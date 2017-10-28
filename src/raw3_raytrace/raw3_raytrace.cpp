// Example raw3_raytrace
// - Simple demonstration of raytracing/pathtracing without any acceleration techniques
// - Casts rays from camera space into scene and recursively traces reflections/refractions
// - Materials are extended to support simple specular reflections and transparency with refraction index

#include <iostream>
#include <ppgso/ppgso.h>
#include <mutex>
#include <iomanip>
#include <atomic>
#include <memory>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <tiny_obj_loader.h>

using namespace std;
using namespace glm;
using namespace ppgso;

// Global constants
constexpr double INF = numeric_limits<double>::max();       // Will be used for infinity
constexpr double EPS = numeric_limits<double>::epsilon();   // Numerical epsilon
const double DELTA = sqrt(EPS);

/*!
 * Structure holding origin and direction that represents a ray
 */
struct Ray {
    dvec3 origin, direction;

    /*!
     * Compute a point on the ray
     * @param t Distance from origin
     * @return Point on ray where t is the distance from the origin
     */
    inline dvec3 point(double t) const {
        return origin + direction * t;
    }
};

/*!
 * Material coefficients for diffuse and emission
 */
struct Material {
    dvec3 emission, diffuse;
    double reflectivity;
    double transparency, refractionIndex;
};

/*!
 * Structure to represent a ray to object collision, the Hit structure will contain material surface normal
 */
struct Hit {
    double distance;
    dvec3 point, normal;
    Material material;
};

/*!
 * Constant for collisions that have not hit any object in the scene
 */
const Hit noHit{INF, {0, 0, 0}, {0, 0, 0}, {{0, 0, 0}, {0, 0, 0}, 0, 0, 0}};                           // Delta to use

// Shape interface
struct Shape {
    virtual ~Shape() = default;
    virtual Hit hit(const Ray& ray) const = 0;
};

// Shape with model matrix that can transform any other shape
struct TransformedShape final : public Shape {
    std::unique_ptr<Shape> shape;
    vec3 rotation = {0, 0, 0};
    vec3 scale = {1, 1, 1};
    vec3 position = {0, 0, 0};

    template<typename T>
    TransformedShape(T s) : shape{std::make_unique<T>(std::move(s))} {}

    virtual Hit hit(const Ray& ray) const override {
        // Compute model matrix and inverse
        glm::mat4 matrix = glm::translate(glm::mat4(1.0f), position)
                           * glm::orientate4(rotation)
                           * glm::scale(glm::mat4(1.0f), scale);
        glm::mat4 inverse = glm::inverse(matrix);

        // Transform ray to object space
        Ray transformedRay = { glm::vec3(inverse * glm::vec4{ray.origin, 1.0f}), glm::vec3(inverse * glm::vec4{ray.direction, 0.0f}) };

        // Hit in object space
        auto hit = shape->hit(transformedRay);

        // Transform to world space
        hit.point = glm::vec3(matrix * glm::vec4{hit.point, 1.0f});
        hit.normal = glm::normalize(glm::vec3(matrix * glm::vec4{hit.normal, 0.0f}));

        return hit;
//        return shape->hit(ray);
    }

};

/*!
 * Structure representing a sphere which is defined by its center position, radius and material
 */
struct Sphere : Shape {
    double radius;
    dvec3 center;
    Material material;

    Sphere(double radius, dvec3 center, Material m) {
        this->radius = radius;
        this->center = center;
        this->material = m;
    }

    /*!
     * Compute ray to sphere collision
     * @param ray Ray to compute collision against
     * @return Hit structure that represents the collision or noHit.
     */
    inline Hit hit(const Ray &ray) const {
        dvec3 oc = ray.origin - center;
        double a = dot(ray.direction, ray.direction);
        double b = dot(oc, ray.direction);
        double c = dot(oc, oc) - radius * radius;
        double dis = b * b - a * c;

        if (dis > 0) {
            double e = sqrt(dis);
            double t = (-b - e) / a;

            if (t > EPS) {
                dvec3 pt = ray.point(t);
                dvec3 n = normalize(pt - center);
                return {t, pt, n, material};
            }

            t = (-b + e) / a;

            if (t > EPS) {
                dvec3 pt = ray.point(t);
                dvec3 n = normalize(pt - center);
                return {t, pt, n, material};
            }
        }
        return noHit;
    }
};

/*!
 * Face structure to hold three vertices that form a triangle/face
 */
struct Triangle : Shape {
    dvec3 v0, v1, v2;
    Material material;

    Triangle(dvec3 v0, dvec3 v1, dvec3 v2, Material material) {
        this->v0 = v0;
        this->v1 = v1;
        this->v2 = v2;
        this->material = material;
    }

    inline Hit hit(const Ray &ray) const override {

        // compute plane's normal
        dvec3 v0v1 = v1 - v0;
        dvec3 v0v2 = v2 - v0;
        // no need to normalize
        dvec3 N = cross(v0v1,v0v2); // N
        float area2 = N.length();

        // Step 1: finding P

        // check if ray and plane are parallel ?
        float NdotRayDirection = dot(N, ray.direction);
        if (fabs(NdotRayDirection) < EPS) // almost 0
            return noHit; // they are parallel so they don't intersect !

        // compute d parameter using equation 2
        float d = dot(N, v0);

        // compute t (equation 3)
        float t = (-1 * (dot(N, ray.origin) + d)) / NdotRayDirection;
        // check if the triangle is in behind the ray
        if (t < 0)
            return noHit; // the triangle is behind

        // compute the intersection point using equation 1
        dvec3 P = ray.point(t);

        // Step 2: inside-outside test
        dvec3 C; // vector perpendicular to triangle's plane

        // edge 0
        dvec3 edge0 = v1 - v0;
        dvec3 vp0 = P - v0;
        C = cross(edge0,vp0);
        if (dot(N, C) < 0)
            return noHit; // P is on the right side

        // edge 1
        dvec3 edge1 = v2 - v1;
        dvec3 vp1 = P - v1;
        C = cross(edge1,vp1);
        if (dot(N, C) < 0)
            return noHit; // P is on the right side

        // edge 2
        dvec3 edge2 = v0 - v2;
        dvec3 vp2 = P - v2;
        C = cross(edge2,vp2);
        if (dot(N, C) < 0)
            return noHit; // P is on the right side;

        N = normalize(N);

        Hit hit = { t, P, N, material }; // this ray hits the triangle
        return hit;
    }
};

struct MyMesh : Shape {
    vector<Triangle> triangles;
    vec3 rotation = {0, -5, 5};
    vec3 scale = {40, 40, 40};
    vec3 position = {0, 0, 0};

    inline Hit hit(const Ray &ray) const override {
        // Compute model matrix and inverse
        glm::mat4 matrix = glm::translate(glm::mat4(1.0f), position)
                           * glm::orientate4(rotation)
                           * glm::scale(glm::mat4(1.0f), scale);
        glm::mat4 inverse = glm::inverse(matrix);

        // Transform ray to object space
        Ray transformedRay = { glm::vec3(inverse * glm::vec4{ray.origin, 1.0f}), glm::vec3(inverse * glm::vec4{ray.direction, 0.0f}) };

        auto hit = noHit;
        for (auto &triangle : triangles)
        {
            auto lh = triangle.hit(transformedRay);
            if (lh.distance < hit.distance) {
                hit = lh;
            }
        }

        // Transform to world space
        hit.point = glm::vec3(matrix * glm::vec4{hit.point, 1.0f});
        hit.normal = glm::normalize(glm::vec3(matrix * glm::vec4{hit.normal, 0.0f}));

        return hit;
    }
};

/*!
 * Load Wavefront obj file data as vector of faces for simplicity
 * @return vector of Faces that can be rendered
 */
MyMesh loadObjFile(const string filename) {
    // Using tiny obj loader from ppgso lib
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string err = tinyobj::LoadObj(shapes, materials, filename.c_str());

    // Will only convert 1st shape to Faces
    auto &mesh = shapes[0].mesh;

    // Collect data in vectors
    vector<vec3> positions;
    for (int i = 0; i < (int) mesh.positions.size() / 3; ++i)
        positions.emplace_back(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);

    // Fill the vector of Faces with data
    MyMesh triangles;
    for (int i = 0; i < (int)(mesh.indices.size() / 3); i++) {
        triangles.triangles.push_back({
                                              positions[mesh.indices[i * 3]],
                                              positions[mesh.indices[i * 3 + 1]],
                                              positions[mesh.indices[i * 3 + 2]],
                                              {                                           // material
                                                      {1, 0, 0},                              // emmision
                                                      {1, 0, 0},                              // difuse
                                                      0,
                                                      0,
                                                      1
                                              }
                                      });
    }
    return triangles;
}

/*!
 * Structure representing a simple camera that is composed on position, up, back and right vectors
 */
struct Camera {
    dvec3 position, back, up, right;

    /*!
     * Generate a new Ray for the given viewport size and position
     * @param x Horizontal position in the viewport
     * @param y Vertical position in the viewport
     * @param width Width of the viewport
     * @param height Height of the viewport
     * @return Ray for the giver viewport position with small random deviation applied to support multi-sampling
     */
    Ray generateRay(int x, int y, int width, int height) const {
        // Camera deltas
        dvec3 vdu = 2.0 * right / (double) width;
        dvec3 vdv = 2.0 * -up / (double) height;

        Ray ray;
        ray.origin = position;
        ray.direction = -back
                        + vdu * ((double) (-width / 2 + x) + linearRand(0.0, 1.0))
                        + vdv * ((double) (-height / 2 + y) + linearRand(0.0, 1.0));
        ray.direction = normalize(ray.direction);
        return ray;
    }
};

/*!
 * Generate a normalized vector that sits on the surface of a half-sphere which is defined using a normal. Used to generate random diffuse reflections.
 * @param normal Normal that defines the dome/half-sphere direction
 * @return Random 3D vector on the dome surface
 */
inline dvec3 RandomDome(const dvec3 &normal) {
    double d;
    dvec3 p;

    do {
        p = sphericalRand(1.0);
        d = dot(p, normal);
    } while (d < 0);

    return p;
}

/*!
 * Structure to represent the scene/world to render
 */
struct World {
    Camera camera;
    vector<Sphere> spheres;
    vector<MyMesh> meshes;
    mutable std::atomic<unsigned long> samplesCounter = {0};
    mutable std::atomic<unsigned long> raysCounter = {0};
    mutable std::atomic<unsigned long> totalCounter = {0};

    /*!
     * Compute ray to object collision with any object in the world
     * @param ray Ray to trace collisions for
     * @return Hit or noHit structure which indicates the material and distance the ray has collided with
     */
    inline Hit cast(const Ray &ray) const {
        Hit hit = noHit;
        for (auto &sphere : spheres) {
            auto lh = sphere.hit(ray);

            if (lh.distance < hit.distance) {
                hit = lh;
            }
        }
        for (auto &mesh : meshes) {
            auto lh = mesh.hit(ray);
            if (lh.distance < hit.distance) {
                hit = lh;
            }
        }
        return hit;
    }

    /*!
     * Trace a ray as it collides with objects in the world
     * @param ray Ray to trace
     * @param depth Maximum number of collisions to trace
     * @return Color representing the accumulated lighting for each ray collision
     */
    inline dvec3 trace(const Ray &ray, unsigned int depth) const {
        if (depth == 0) return {0, 0, 0};

        const Hit hit = cast(ray);

        // No hit
        if (hit.distance == INF) return {0, 0, 0};

        // Emission
        dvec3 color = hit.material.emission;

        // Decide to reflect or refract using linear random
        if (linearRand(0.0f, 1.0f) < hit.material.transparency) {
            // Flip normal if the ray is "inside" a sphere
            dvec3 normal = dot(ray.direction, hit.normal) < 0 ? hit.normal : -hit.normal;
            // Reverse the refraction index as well
            double r_index = dot(ray.direction, hit.normal) < 0 ? 1 / hit.material.refractionIndex
                                                                : hit.material.refractionIndex;

            // Prepare refraction ray
            dvec3 refraction = refract(ray.direction, normal, r_index);
            Ray refractionRay{hit.point - normal * DELTA, refraction};
            // Modulate the refraction color with diffuse color
            dvec3 refractionColor = lerp(hit.material.diffuse, {1, 1, 1}, hit.material.transparency);
            // Trace the ray recursively
            color += refractionColor * trace(refractionRay, depth - 1);
        } else {
            // Calculate reflection
            // Random diffuse reflection
            dvec3 diffuse = RandomDome(hit.normal);
            // Ideal specular reflection
            dvec3 reflection = reflect(ray.direction, hit.normal);
            // Ray that combines reflection direction depending on the material reflectivness
            Ray reflectedRay{hit.point + hit.normal * DELTA, lerp(diffuse, reflection, hit.material.reflectivity)};
            // Reflection color is white for specular reflections, otherwise diffuse color is used
            dvec3 reflectionColor = lerp(hit.material.diffuse, {1, 1, 1}, hit.material.reflectivity);
            // Trace the ray recursively
            color += reflectionColor * trace(reflectedRay, depth - 1);
        }

        raysCounter++;

        return color;
    }

    /*!
     * Render the world to the provided image
     * @param image Image to render to
     */
    void render(Image &image, unsigned int samples, unsigned int depth) const {

        int imageTotal = image.width * image.height;
        clock_t start = clock();
        std::mutex mtx;

        // For each pixel generate rays
        #pragma omp parallel for
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                dvec3 color;

                // Generate multiple samples
                for (unsigned int i = 0; i < samples; ++i) {
                    auto ray = camera.generateRay(x, y, image.width, image.height);
                    color = color + trace(ray, depth);
                    samplesCounter++;
                }
                // Collect the data
                color = clamp(color / (double) samples, 0.0, 1.0);
                image.setPixel(x, y, (float) color.r, (float) color.g, (float) color.b);
            }

            totalCounter += image.width;

            auto prog = clock();
            double t = double(prog - start) / CLOCKS_PER_SEC;
            double perc = round(double(totalCounter) / imageTotal * 10000) / 100;
            double sampCnt = double(samplesCounter) / t;
            double raysCnt = double(raysCounter) / t;

            {
                std::lock_guard<std::mutex> lock(mtx);
                cout << "Progress [" << setw(6) << perc << "%] | "
                     << setw(14) << sampCnt << " samples/sec. | "
                     << setw(14) << raysCnt << " rays/sec.\r";
            }
        }
    }
};

int main() {
    cout << "This will take a while ..." << endl;

    // Image to render to
    Image image{512, 512};

    // World to render
    const World world{
            { // Camera
                    {  0,  0, 25}, // Position
                    {  0,  0,  1}, // Back
                    {  0, .5,  0}, // Up
                    { .5,  0,  0}, // Right
            },
            { // Spheres
//                    {10000, {0, -10010, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Floor
//                    {10000, {-10010, 0, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Left wall
//                    {10000, {10010,  0, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Right wall
//                    {10000, {0, 0, -10010},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Back wall
//                    {10000, {0, 0,  10030},  {{0, 0, 0}, {1, 1, 0}, 0, 0, 0}},           // Front wall (behind camera)
//                    {10000, {0,  10010, 0},  {{1, 1, 1}, {1, 1, 1}, 0, 0, 0}},           // Ceiling and source of light
//                    {2, {-5, -8, 3}, {{0, 0, 0}, {.7, .7, 0}, 1, .95, 1.52}},          // Refractive glass sphere
//                    {4, {0, -6, 0}, {{0, 0, 0}, {.7, .5, .1}, 1, 0, 0}},               // Reflective sphere
//                    {10, {10, 10, -10}, {{0, 0, 0}, {0, 0, 1}, 0, 0, 1.54}},           // Sphere in top right corner
            },
//            {
//                TransformedShape(Triangle{
//                    {0, 0, 0},
//                    {0, 5, 0},
//                    {5, 0, 0},
//                    {{0, 1, 0}, {1, 0, 0}, 1, 0, 0}
//                }),
//                TransformedShape(Triangle{
//                    {0, 0, 0},
//                    {0, 2.5, 2.5},
//                    {5, 0, 0},
//                    {{1, 0, 0}, {1, 0, 0}, 1, 0, 0}
//                })
//            }
            {
                loadObjFile("..\\data\\bunny.obj")
            }
    };


//    TransformedShape shape(bunny);
//    shape.position = position;
//    shape.rotation = rotation;
//    shape.scale = scale;
//    world.meshes.push_back((shape));

    // Render the scene
    world.render(image, 4, 4);

    // Save the result
    image::saveBMP(image, "raw3_raytrace.bmp");

    cout << "Done." << endl;
    return EXIT_SUCCESS;
}
