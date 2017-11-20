// Example raw3_raytrace
// - Simple demonstration of raytracing/pathtracing without any acceleration techniques
// - Casts rays from camera space into scene and recursively traces reflections/refractions
// - Materials are extended to support simple specular reflections and transparency with refraction index

#include <iostream>
#include <ppgso/ppgso.h>
#include <mutex>
#include <iomanip>
#include <atomic>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

using namespace std;
using namespace glm;
using namespace ppgso;

const bool useKD = true;

// Global constants
constexpr double INF = numeric_limits<float>::max();       // Will be used for infinity
constexpr double EPS = numeric_limits<float>::epsilon();   // Numerical epsilon
const double DELTA = (double)sqrt(EPS);

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

class ImageFloat {
public:

    /*!
     * Create new empty image.
     *
     * @param width - Width in pixels.
     * @param height - Height in pixels.
     */
    ImageFloat(int width, int height) : width{width}, height{height} {
        framebuffer.resize((size_t) (width * height));
    }

    /*!
	 * Create new empty image.
	 *
	 * @param width - Width in pixels.
	 * @param height - Height in pixels.
	 */
    ImageFloat(const string fileName) {

        int n;
        auto img = stbi_loadf(fileName.c_str(), &width, &height, &n, 3);

        if (img != NULL) {
            framebuffer.resize((size_t) (width * height));
	        for (int row = 0; row < height; row++) {
		        for (int col = 0; col < width; col++) {
			        framebuffer[col + row * width] = vec3(img[row * width * 3 + 3 * col],
			                                              img[row * width * 3 + 3 * col + 1],
			                                              img[row * width * 3 + 3 * col + 2]);
		        }
	        }
        }
    }

    /*!
	 * Get raw access to the image data.
	 *
	 * @return - Pointer to the raw RGB framebuffer data.
	 */
    std::vector<vec3>& getFramebuffer() {
        return framebuffer;
    }

    /*!
	 * Get single pixel from the framebuffer.
	 *
	 * @param x - X position of the pixel in the framebuffer.
	 * @param y - Y position of the pixel in the framebuffer.
	 * @return - Reference to the pixel.
	 */
    vec3& getPixel(int x, int y) {
        return framebuffer[x+y*width];
    }

    /*!
	 * Set pixel on coordinates x and y
	 * @param x Horizontal coordinate
	 * @param y Vertical coordinate
	 * @param color Pixel color to set
	 */
    void setPixel(int x, int y, const vec3& color) {
        framebuffer[x+y*width] = color;
    }

    /*!
	 * Set pixel on coordinates x and y
	 * @param x Horizontal coordinate
	 * @param y Vertical coordinate
	 * @param r Red channel <0, 255>
	 * @param g Green channel <0, 255>
	 * @param b Blue channel <0, 255>
	 */
    void setPixel(int x, int y, int r, int g, int b) {
        setPixel(x,y,{(uint8_t)r, (uint8_t)g, (uint8_t)b});
    }

    /*!
	 * Set pixel on coordinates x and y
	 * @param x Horizontal coordinate
	 * @param y Vertical coordinate
	 * @param r Red channel <0, 1>
	 * @param g Green channel <0, 1>
	 * @param b Blue channel <0, 1>
	 */
    void setPixel(int x, int y, float r, float g, float b) {
        setPixel(x,y,{(uint8_t) (r * 255), (uint8_t) (g * 255), (uint8_t) (b * 255)});
    }

    /*!
	 * Clear the image using single color
	 * @param color Pixel color to set the image to
	 */
    void clear(const vec3& color = {0,0,0}) {
        framebuffer = vector<vec3>(framebuffer.size(), color);
    }

    int width, height;
private:
    std::vector<vec3> framebuffer;
};

/*!
 * Constant for collisions that have not hit any object in the scene
 */
const Hit noHit{INF, {0, 0, 0}, {0, 0, 0}, {{0, 0, 0}, {0, 0, 0}, .0, .0, .0}};                           // Delta to use

// Shape interface
struct Shape {
    virtual ~Shape() = default;
    virtual Hit hit(const Ray& ray) const = 0;
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

    Triangle(vec3 v0, dvec3 v1, dvec3 v2, Material material) {
        this->v0 = v0;
        this->v1 = v1;
        this->v2 = v2;
        this->material = material;
    }

    inline Hit hit(const Ray &ray) const {
        double t, u, v;

        dvec3 v0v1 = v1 - v0;
        dvec3 v0v2 = v2 - v0;
        dvec3 pvec = cross(ray.direction, v0v2);
        double det = dot(v0v1, pvec);

        if (fabs(det) < EPS) return noHit;

        double invDet = 1 / det;

        dvec3 tvec = ray.origin - v0;

        u = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return noHit;

        dvec3 qvec = cross(tvec, v0v1);
        v = dot(ray.direction, qvec) * invDet;
        if (v < 0 || u + v > 1) return noHit;

        t = -1 * dot(v0v2, qvec) * invDet;

        return {t, ray.point(t), normalize(cross(v0v1, v0v2)), material};
    }
};

struct Box {
    dvec3  x1, x2;
    Material material;

    Box() {
        x1 = { 0, 0, 0 };
        x2 = { 0, 0, 0 };
        material = {                                      // material
            {1.0, 0.0, 0.0},                              // emmision
            {0.8, 0.2, 0.0},                              // difuse
            0.0,
            0.0,
            0.0
        };
    }

    void expand(const Triangle &triangle) {
        double minx = std::min(std::min(triangle.v0.x, triangle.v1.x), triangle.v2.x);
        double miny = std::min(std::min(triangle.v0.y, triangle.v1.y), triangle.v2.y);
        double minz = std::min(std::min(triangle.v0.z, triangle.v1.z), triangle.v2.z);

        double maxx = std::max(std::max(triangle.v0.x, triangle.v1.x), triangle.v2.x);
        double maxy = std::max(std::max(triangle.v0.y, triangle.v1.y), triangle.v2.y);
        double maxz = std::max(std::max(triangle.v0.z, triangle.v1.z), triangle.v2.z);

        if (x1.x > minx)
            x1.x = minx;
        if (x1.y > miny)
            x1.y = miny;
        if (x1.z > minz)
            x1.z = minz;

        if (x2.x < maxx)
            x2.x = maxx;
        if (x2.y < maxy)
            x2.y = maxy;
        if (x2.z < maxz)
            x2.z = maxz;
    }

    inline Hit intersect(const Ray &ray) const
    {
        try {
            dvec3 n1 = (x1 - ray.origin) / ray.direction;
            dvec3 f1 = (x2 - ray.origin) / ray.direction;
            auto n = glm::min(n1, f1);
            auto f = glm::max(n1, f1);
            double t0 = std::max(std::max(n.x, n.y), n.z);
            double t1 = std::min(std::min(f.x, f.y), f.z);

            dvec3 point = ray.point(t0);
            dvec3 normal;
            if (point.x < x1.x + EPS)
                normal = {-1, 0, 0};
            else if (point.x > x2.x - EPS)
                normal = {1, 0, 0};
            else if (point.y < x1.y + EPS)
                normal = {0, -1, 0};
            else if (point.y > x2.y - EPS)
                normal = {0, 1, 0};
            else if (point.z < x1.z + EPS)
                normal = {0, 0, -1};
            else if (point.z > x2.z - EPS)
                normal = {0, 0, 1};
            else
                normal = {0, 1, 0};

            if (t0 > 0.0 && t0 < t1) {
                return Hit{t0, point, normal, material};
            }
        }
        catch (exception ex) {

        }

        return noHit;
    }
};

struct MyMesh : Shape {
    Box bbox = Box();
    vector<Triangle> triangles = vector<Triangle>();
    std::shared_ptr<MyMesh> leftNode = nullptr;
    std::shared_ptr<MyMesh> rightNode = nullptr;
    double midPoint = 0.0f;
    int axis = -1;

    inline Hit hit(const Ray &ray) const override {
        auto hit = noHit;
        Hit h;
        if (!useKD) {
            for (auto& triangle : triangles) {
                auto lh = triangle.hit(ray);
                if (lh.distance < hit.distance) {
                    hit = lh;
                }
            }
        } else {
            if (leftNode != nullptr) {
                h = leftNode->bbox.intersect(ray);
                if (h.distance < hit.distance)
                    hit = leftNode->hit(ray);
            }

            if (rightNode != nullptr) {
                h = rightNode->hit(ray);
                if (h.distance < hit.distance)
                    hit = rightNode->hit(ray);
            }

            if (!triangles.empty())
                for (auto& triangle : triangles) {
                    h = triangle.hit(ray);
                    if (h.distance < hit.distance)
                        hit = h;
                }
        }

        return hit;
    }

    inline void build(vector<Triangle> &triangles, int depth) {

        axis = depth % 3;
        bbox = Box();
        for (auto& triangle : triangles)
            bbox.expand(triangle);

        if (triangles.size() <= 8) {

            for (auto &t : triangles)
                this->triangles.emplace_back(t);

            return;
        }

        midPoint = getMidPoint(triangles);
        vector<Triangle> left;
        vector<Triangle> right;
        for (auto& triangle : triangles) {
            if (axis == 0)
                std::max(std::max(triangle.v0.x, triangle.v1.x), triangle.v2.x) < midPoint ?
                left.emplace_back(triangle) : right.emplace_back(triangle);
            else if (axis == 1)
                std::max(std::max(triangle.v0.y, triangle.v1.y), triangle.v2.y) < midPoint ?
                left.emplace_back(triangle) : right.emplace_back(triangle);
            else
                std::max(std::max(triangle.v0.z, triangle.v1.z), triangle.v2.z) < midPoint ?
                left.emplace_back(triangle) : right.emplace_back(triangle);
        }

//        leftNode = new MyMesh();
//        rightNode = new MyMesh();
        leftNode = std::make_shared<MyMesh>();
        rightNode = std::make_shared<MyMesh>();

        leftNode->build(left, depth + 1);
        rightNode->build(right, depth + 1);
    }

    inline double getMidPoint(vector<Triangle> triangles) {
        double res = 0;
        vector<float> values;
        if (axis == 0) { // x
            for (auto& triangle : triangles)
                values.emplace_back(std::max(std::max(triangle.v0.x,triangle.v1.x),triangle.v2.x));
        } else if (axis == 1) { // y
            for (auto& triangle : triangles)
                values.emplace_back(std::max(std::max(triangle.v0.y,triangle.v1.y),triangle.v2.y));
        } else { // z
            for (auto& triangle : triangles)
                values.emplace_back(std::max(std::max(triangle.v0.z,triangle.v1.z),triangle.v2.z));
        }

        sort(values.begin(), values.end());

        auto size = values.size();
        if (size  % 2 == 0)
            res = (values[size / 2 - 1] + values[size / 2]) / 2;
        else
            res = values[size / 2];

        return res;
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
        positions.emplace_back(
            (mesh.positions[3 * i] * 50.0f + 1),
            (mesh.positions[3 * i + 1] * 50.0f - 9),
            (mesh.positions[3 * i + 2] * 50.0f)
        );

    // Fill the vector of Faces with data
    vector<Triangle> triangles;
    for (int i = 0; i < (int)(mesh.indices.size() / 3); i++) {
        triangles.emplace_back(Triangle(
            positions[mesh.indices[i * 3]],
            positions[mesh.indices[i * 3 + 1]],
            positions[mesh.indices[i * 3 + 2]],
            {                                           // material
                {0.0, 0.0, 0.0},                              // emmision
                {0.8, 0.2, 0.0},                              // difuse
                0.0,
                0.0,
                0.0
            }
        ));
    }

    MyMesh res = MyMesh();

    if (useKD)
        res.build(triangles, 0);
    else
        res.triangles = triangles;

    return res;
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
    Ray generateRay(int x, int y, int width, int height, int fstop, int distance) const {
        // Camera deltas
        dvec3 vdu = 2.0 * right / (double) width;
        dvec3 vdv = 2.0 * -up / (double) height;

        Ray ray;
        ray.origin = position;
        ray.direction = -back
                        + vdu * ((double) (-width / 2 + x) + linearRand(0.0, 1.0))
                        + vdv * ((double) (-height / 2 + y) + linearRand(0.0, 1.0));
        ray.direction = normalize(ray.direction);

        dvec3 focusPoint = ray.point(distance);
        ray.origin += dvec3 {
                glm::linearRand(-1.0f, 1.0f) / (float)fstop,
                glm::linearRand(-1.0f, 1.0f) / (float)fstop,
                0
        };
        ray.direction = normalize(focusPoint - ray.origin);

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
    ImageFloat map;
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

    inline dvec3 getMapColor(const Ray &ray) {
        double x = floor((atan2(ray.direction.z, ray.direction.x) + PI) / (2 * PI) * map.width);
        double y = floor(acos(ray.direction.y) / PI * map.height);
        auto px = map.getPixel(x, y);
//	    cout << "[" << x << ", " << y << "] [" << px.x << ", " << px.y << ", " << px.z <<"]" << endl;
        return dvec3(px.x, px.y, px.z);
    }

    /*!
     * Trace a ray as it collides with objects in the world
     * @param ray Ray to trace
     * @param depth Maximum number of collisions to trace
     * @return Color representing the accumulated lighting for each ray collision
     */
    inline dvec3 trace(const Ray &ray, unsigned int depth) {
        if (depth == 0) return getMapColor(ray);

        const Hit hit = cast(ray);

        // No hit
        if (hit.distance == INF) return getMapColor(ray);

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
    void render(ImageFloat &image, unsigned int samples, unsigned int depth) {

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
                    auto ray = camera.generateRay(x, y, image.width, image.height, 2, 35);
                    color = color + trace(ray, depth);
                    samplesCounter++;
                }
                // Collect the data
                color = clamp(color / (double) samples, 0.0, 1.0);
                image.setPixel(x, y, (float) color.r, (float) color.g, (float) color.b);
            }

            totalCounter += image.width;

            auto prog = clock();
            double t = float(prog - start) / CLOCKS_PER_SEC;
            double perc = round(float(totalCounter) / imageTotal * 10000) / 100;
            double sampCnt = float(samplesCounter) / t;
            double raysCnt = float(raysCounter) / t;

            {
                std::lock_guard<std::mutex> lock(mtx);
                cout << "Progress [" << setw(6) << perc << "%] | "
                     << setw(14) << sampCnt << " samples/sec. | "
                     << setw(14) << raysCnt << " rays/sec.\r";
            }
        }

        clock_t end = clock();
        cout << endl << "Execution takes: " << double(end - start) / CLOCKS_PER_SEC << " seconds" << endl;
    }
};

int main() {
    cout << "This will take a while ..." << endl;

    // Image to render to
    ImageFloat image{512, 512};

    // World to render
    World world{
        { // Camera
            {  0,  0, 50}, // Position
            {  0,  0,  1}, // Back
            {  0, .5,  0}, // Up
            { .5,  0,  0} // Right
        },
        { // Spheres
//                    {10000, {0, -10010, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Floor
//                    {10000, {-10010, 0, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Left wall
//                    {10000, {10010,  0, 0},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Right wall
//                    {10000, {0, 0, -10010},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Back wall
//                    {10000, {0, 0,  10030},  {{0, 0, 0}, {1, 1, 1}, 0, 0, 0}},           // Front wall (behind camera)
//                    {10000, {0,  10010, 0},  {{1, 1, 1}, {1, 1, 1}, 0, 0, 0}},           // Ceiling and source of light
//                    {2, {-5, -8, 3}, {{0, 0, 0}, {.7, .7, 0}, 1, .95, 1.52}},          // Refractive glass sphere
                    { 4, {10, 10, 15}, { {0, 0, 0}, {.7, .5, .1}, 1, 0, 0 } },
                    { 4, {0, 0, 0}, { {0, 0, 0}, {.7, .5, .1}, 1, 0, 0 } },
                    { 4, {-10, -10, -15}, { {0, 0, 0}, {.7, .5, .1}, 1, 0, 0 } }// Reflective sphere
//                    {10, {10, 10, -10}, {{0, 0, 0}, {0, 0, 1}, 0, 0, 1.54}},           // Sphere in top right corner
        },
//            {
//                {
//                    {-8.656, 14.249, 0.843},
//                    {-8.969, 13.971, 1.377},
//                    {-8.949, 14.392, 1.248},
//                    {{1, 0, 0}, {1, 0, 0}, 1, 0, 0}
//                }
//            }
//        { loadObjFile("..\\data\\bunny.obj") },
        {},
        ImageFloat("..\\data\\mapa2.jpg")
    };

    // Render the scene
    world.render(image, 256, 4);

    Image img(image.width, image.height);

    // gamma correction
    float A = 1.0;
    float y = 1.2;
    for ( int row = 0; row < image.height; row++)
        for (int col = 0; col < image.width; col++)
        {
            auto p = image.getPixel(col, row);
            int r = std::min((int)round(powf(p.r, y) * A ), 255);
            int g = std::min((int)round(powf(p.g, y) * A ), 255);
            int b = std::min((int)round(powf(p.b, y) * A ), 255);
            img.setPixel(col, row, r, g, b);
        }


    // Save the result
    image::saveBMP(img, "raw3_raytrace.bmp");

    cout << "Done." << endl;
    return EXIT_SUCCESS;
}
