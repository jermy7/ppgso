#pragma once

#include <chrono>
#include <future>

#include <ppgso/ppgso.h>

#include "ray.h"
#include "shape.h"
#include "camera.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

// Parallel FOR template
template<typename F>
void parallel_for(int begin, int end, F fn, int fragment_size = 0) {
    int fragment_count = std::thread::hardware_concurrency();
    int length = end - begin;

    if (fragment_size == 0) {
        fragment_size = length / fragment_count;
    }

    if (length <= fragment_size) {
        for (int i = begin; i < end; i++) {
            fn(i);
        }
        return;
    }

    int mid = (begin + end) / 2;
    auto handle = async(std::launch::async, parallel_for<F>, begin, mid, fn, fragment_size);
    parallel_for(mid, end, fn, fragment_size);
    handle.get();
}

class Renderer {
    // Statistical reports
    mutable std::atomic<int> current_rows{0};
    mutable std::atomic<int> current_samples{0};
    mutable std::atomic<int> current_rays{0};

    int mapWidth, mapHeight;
    std::vector<glm::vec3> environmentMap;

    glm::vec3 getPixel(int x, int y) const {
        return environmentMap[y * mapWidth + x];
    }

    float ggxGeometry(Vector e, Vector n, float alpha) const {
        float en = glm::dot(e, n);
        float alpha2 = powf(alpha, 2);
        return (2 * en) / (en + sqrtf(alpha2 + (1 + alpha2) * powf(en, 2)));
    }

    float smithGeometry(Direction v, Direction n, float k) const {
        float nv = fabsf(glm::dot(n, v));
        return nv / (nv * (1 - k) + k);
    }

    Color schlickAppriximation(Color f0, Vector v, Vector half) const {
        float vh = glm::dot(v, half);
//        return f0 + (1.0f - f0) * powf(2, (-5.55473f * vh - 6.98316f) * vh);
        return f0 + (1.0f - f0) * powf((1.0f - vh), 5);
    }

    Vector importanceSamplerGgx(glm::vec2 random, float roughness, Vector normal) const {
        float alpha = powf(roughness, 2);

        float phi = 2 * (float)M_PI * random.x;
        float cosTheta = sqrtf((1 - random.y) / (1 + (powf(alpha, 2) - 1) * random.y));
        float sinTheta = sqrtf(1 - powf(cosTheta, 2));

        Vector halfVector;
        halfVector.x = sinTheta * cosf(phi);
        halfVector.y = sinTheta * sinf(phi);
        halfVector.z = cosTheta;

        Vector upVector = (fabsf(normal.z) < 0.999f) ? Vector(0.0f, 0.0f, 1.0f) : Vector(1.0f, 0.0f, 0.0f);
        Vector tangentX = glm::normalize(glm::cross(upVector, normal));
        Vector tangentY = glm::cross(normal, tangentX);

        return tangentX * halfVector.x + tangentY * halfVector.y + normal * halfVector.z;
    }

public:
    Camera camera;
    std::vector<std::unique_ptr<Shape>> scene;

    struct pixel {
        int samples = 0;
        Color color;

        inline void add(Color sample) {
            samples++;
            color = color + (sample - color) / (float) samples;
        }
    };

    std::vector<pixel> samples;
    int width;
    int height;

    inline Renderer(int width, int height, char *environmentMapFile = NULL) : width{width}, height{height},
                                                                              samples{(size_t) width * height} {
        int n;

        stbi_hdr_to_ldr_gamma(2.2f);
        stbi_hdr_to_ldr_scale(1.0f);
        float *data = stbi_loadf(environmentMapFile, &mapWidth, &mapHeight, &n, 3);
        if (data != NULL) {
            for (int i = 0; i < mapHeight; i++) {
                for (int j = 0; j < mapWidth; j++) {
                    environmentMap.push_back({
                                                     data[(i * mapWidth + j) * n + 0],
                                                     data[(i * mapWidth + j) * n + 1],
                                                     data[(i * mapWidth + j) * n + 2]
                                             });
                }
            }
        }
        stbi_image_free(data);
    }

    inline void render(int depth = 5) {
        auto rendering = async(
                std::launch::async,
                [&]() {
                    parallel_for(
                            0,
                            height,
                            // Lambda function
                            [&](int y) {
                                for (int x = 0; x < width; ++x) {
                                    auto ray = camera.generateRay(x, y, width, height, 100, 30);
                                    Color sample = trace(ray, depth);
                                    samples[width * y + x].add(sample);
                                    this->current_samples++;
                                }
                                this->current_rows++;
                            }
                    );
                }
        );

        std::chrono::milliseconds span(1000);
        while (rendering.wait_for(span) == std::future_status::timeout) {
            int progress = (int) (((float) current_rows / (float) height) * 100.0f) % 100;
            std::cout << "Progress: " << progress << "%.\n"
                      << "Samples per second: " << current_samples << ".\n"
                      << "Rays per second: " << current_rays << ".\n"
                      << "\n";
            current_samples = 0;
            current_rays = 0;
        }


//    // Render the scene
//    #pragma omp parallel for
//    for (int y = 0; y < height; ++y) {
//      for (int x = 0; x < width; ++x) {
//        auto ray = camera.generateRay(x, y, width, height);
//        Color sample = trace(ray, depth);
//        samples[width*y+x].add(sample);
//        this->current_samples ++;
//      }
//      this->current_rows ++;
//    }

    }


    /*!
     * Compute ray to object collision with any object in the world
     * @param ray Ray to trace collisions for
     * @return Hit or noHit structure which indicates the material and distance the ray has collided with
     */
    inline Hit cast(const Ray &ray) const {
        Hit hit = noHit;
        for (auto &shape : scene) {
            auto lh = shape->intersect(ray);

            if (lh.distance < hit.distance && lh.distance > DELTA) {
                hit = lh;
            }
        }
        this->current_rays++;
        return hit;
    }

    /*!
    * Trace a ray as it collides with objects in the world
    * @param ray Ray to trace
    * @param depth Maximum number of collisions to trace
    * @return Color representing the accumulated lighting for each ray collision
    */
    inline Color trace(const Ray &ray, unsigned int depth) const {
        if (depth == 0) return {0, 0, 0};

        const Hit hit = cast(ray);

        // No hit
        if (hit.distance >= INF) {
            float yaw = atan2f(ray.direction.z, ray.direction.x) / (2 * (float)M_PI) + 0.5f;
            float pitch = acosf(ray.direction.y) / (float)M_PI;

            auto xMapCoord = (int)(yaw * mapWidth);
            xMapCoord = (xMapCoord >= mapWidth) ? mapWidth - 1 : xMapCoord;
            auto yMapCoord = (int)(pitch * mapHeight);
            yMapCoord = (yMapCoord >= mapHeight) ? mapHeight - 1 : yMapCoord;

            return getPixel(xMapCoord, yMapCoord);
        }

        // Scene can have artificial lights
        if (hit.material->getIsLight()) {
            return hit.material->getLight();
        }

        Vector normal = hit.normal;
        Vector viewVector = -ray.direction;

        float metalness = hit.material->getMetalness(normal);

        if (glm::linearRand(0.0f, 1.0f) <= metalness) {
            Color albedo = hit.material->getAlbedo(normal);
            float roughness = hit.material->getRoughness(normal);

            Vector halfVector = importanceSamplerGgx(glm::linearRand(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f)), roughness, normal);
            Vector lightVector = 2.0f * glm::dot(viewVector, halfVector) * halfVector - viewVector;

            float nv = glm::clamp(glm::dot(normal, viewVector), 0.0f, 1.0f);
            float vh = glm::clamp(glm::dot(viewVector, halfVector), 0.0f, 1.0f);
            float nh = glm::clamp(glm::dot(normal, halfVector), 0.0f, 1.0f);

            // --- GEOMETRIC TERM ---
            float k = (powf(roughness, 2)) / 2;
            float geometry = smithGeometry(viewVector, normal, k) * smithGeometry(lightVector, normal, k);

            // --- FRESNEL ---
            Color fresnelSpecular = schlickAppriximation(hit.material->getSpecular(), viewVector, halfVector);

            // --- BRDF ---
            Color brdf = fresnelSpecular * geometry * vh / (nh * nv);

            // Ray of reflection
            Ray reflectedRay{hit.position + normal * DELTA, lightVector};
            return brdf * trace(reflectedRay, depth - 1);
        } else {
            // Fresnel
            float f0 = powf((1.0f - hit.material->getIor()) / (1.0f + hit.material->getIor()), 2);
            Color diffuseFresnel = schlickAppriximation({f0, f0, f0}, viewVector, normal);

            if (glm::linearRand(0.0f, 1.0f) < diffuseFresnel.x) {
                // --- Fresnel ---
                // Ideal specular reflection
                Direction reflection = -reflect(viewVector, normal);
                // Ray of reflection
                Ray reflectedRay{hit.position + normal * DELTA, reflection};
                return trace(reflectedRay, depth - 1);
            } else {
                // --- Refraction ---
                Color albedo = hit.material->getAlbedo(normal);
                float transparency = hit.material->getTransparency();
                if (glm::linearRand(0.0f, 1.0f) < transparency) {
                    // Flip normal if the ray is "inside" a sphere
                    normal = dot(ray.direction, normal) < 0 ? normal : -normal;

                    // Reverse the refraction index as well
                    float r_index = dot(ray.direction, hit.normal) < 0 ? 1/hit.material->getIor() : hit.material->getIor();
                    float eta = powf((r_index - 1.0f)/(r_index + 1.0f), 2);

                    // Total internal refraction
                    float ddn = glm::dot(ray.direction, hit.normal);
                    float cos2t = 1 - eta * eta * (1 - ddn * ddn);
                    if(cos2t < 0) {
                        // Ideal specular reflection
                        Direction reflection = reflect(ray.direction, normal);
                        // Ray of reflection
                        Ray reflectedRay{hit.position, reflection};
                        return albedo + trace(reflectedRay, depth - 1);
                    }

                    Direction refraction = refract(ray.direction, normal, eta);
                    Ray refractRay = Ray{hit.position + normal * DELTA, refraction};
                    return trace(refractRay, depth - 1);

//                    return dvec3{1,1,1} * trace(Ray{hit.point - N * DELTA, refract(V, N, eta)}, depth - 1, false);

//                    // --- Refraction ---
//                    const float refractionIndex = hit.material->getIor();
//
//                    // Ideal specular reflection
//                    Direction reflection = reflect(ray.direction, hit.normal);
//                    // Ray of reflection
//                    Ray reflectedRay{hit.position, reflection};
//
//                    // Flip normal if the ray is "inside" a sphere
//                    Direction normal = dot(ray.direction, hit.normal) < 0 ? hit.normal : -hit.normal;
//                    // Reverse the refraction index as well
//                    float r_index = dot(ray.direction, hit.normal) < 0 ? 1/refractionIndex : refractionIndex;
//
//                    // Total internal refraction
//                    float ddn = dot(ray.direction, hit.normal);
//                    float cos2t = 1-r_index*r_index*(1-ddn*ddn);
//                    if(cos2t < 0)
//                        return albedo + trace(reflectedRay, depth - 1);
//
//                    // Prepare refraction ray
//                    Direction refraction = refract(ray.direction, normal, r_index);
//                    Ray refractionRay{hit.position, refraction};
//                    // Trace the ray recursively
//                    return trace(refractionRay, depth - 1);
                } else {
                    // --- Diffusion --
                    // Random diffuse reflection
                    Direction diffuse = CosineSampleHemisphere(normal);
                    // Random diffuse ray
                    Ray diffuseRay{hit.position + normal * DELTA, diffuse};
                    // Trace the ray recursively
                    return albedo * trace(diffuseRay, depth - 1);
//                    return albedo / (float)M_PI;
                }
            }
        }
    }

    template<typename T>
    void add(T shape) {
        scene.emplace_back(std::make_unique<T>(std::move(shape)));
    }
};
