#pragma once
// Example gl3_animate
// - Demonstrates the use of a dynamically generated texture content on the CPU
// - Displays the generated content as texture on a quad using OpenGL
// - Basic animation achieved by incrementing a parameter used in the image generation

#include <iostream>
#include <thread>
#include <glm/glm.hpp>

#include <ppgso/ppgso.h>

#include <shaders/texture_vert_glsl.h>
#include <shaders/texture_frag_glsl.h>

#include "box.h"
#include "meshobject.h"
#include "renderer.h"
#include "sphere.h"

typedef enum ToneMapping {
  LINEAR,
  GLOBAL_REINHARD,
  UNCHARTED_2,
};

using namespace std;
using namespace glm;
using namespace ppgso;

const unsigned int SIZE = 512;
ToneMapping TONE_MAPPING;

/*!
 * Load Wavefront obj file data as vector of faces for simplicity
 * @return vector of Faces that can be rendered
 */
vector<Triangle> loadObjFile(const string filename) {
  // Using tiny obj loader from ppgso lib
  vector<tinyobj::shape_t> shapes;
  vector<tinyobj::material_t> materials;
  string err = tinyobj::LoadObj(shapes, materials, filename.c_str());

  // Will only convert 1st shape to Faces
  auto &mesh = shapes[0].mesh;

  // Collect data in vectors
  vector<vec3> positions;
  for (int i = 0; i < (int) mesh.positions.size() / 3; ++i) {
    positions.emplace_back(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);
  }

  // Fill the vector of Faces with data
  Material material("texture1/albedo.jpg", "texture1/metalness.jpg", "texture1/roughness.jpg");
  vector<Triangle> triangles;
  for (int i = 0; i < (int) (mesh.indices.size() / 3); i++) {
    vec3 v1 = {positions[mesh.indices[i * 3]].x * 75 + 2, positions[mesh.indices[i * 3]].y * 75 - 10, positions[mesh.indices[i * 3]].z * 75};
    vec3 v2 = {positions[mesh.indices[i * 3 + 1]].x * 75 + 2, positions[mesh.indices[i * 3 + 1]].y * 75 - 10, positions[mesh.indices[i * 3 + 1]].z * 75};
    vec3 v3 = {positions[mesh.indices[i * 3 + 2]].x * 75 + 2, positions[mesh.indices[i * 3 + 2]].y * 75 - 10, positions[mesh.indices[i * 3 + 2]].z * 75};

    auto triangle = Triangle(v1, v2, v3, material);
    triangles.emplace_back(move(triangle));
  }
  return triangles;
}

/*!
 * Custom window that will update its contents to create animation
 */
class PathTracerWindow : public Window {
private:
  // Create shading program from included shader sources
  Shader program = {texture_vert_glsl, texture_frag_glsl};

  // Load a quad mesh
  Mesh quad = {"quad.obj"};

  // Our Path Tracer
  Renderer renderer;

  Texture framebuffer;

  // Uncharted Tone Mapper constants
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;
  const float W = 11.2;
  const float ExposureBias = 2.0f;

  Color Uncharted2Tonemap(Color x) {
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
  }

public:
  /*!
   * Construct a new Window and initialize shader uniform variables
   */
  PathTracerWindow() : Window{"pt_pathtracer", SIZE, SIZE}, renderer{SIZE, SIZE, "../scene/maps/map.hdr"}, framebuffer{SIZE, SIZE} {
    // Prepare the scene
    auto& scene = renderer.scene;

    renderer.camera.position = {0,0,15}; // 15

//      renderer.add(Sphere(100000, Position(0,-100015, 0), Material("../data/orange.png", nullptr, nullptr,true)));

    auto light = Material(Color( 1.0f, 1.0f, 1.0f ));
    renderer.add(Sphere( 3, Position(-5,0,-5), light ));

    auto diffuse = Material("../scene/materials/Lava/color.jpg", nullptr, "../scene/materials/Lava/gloss.jpg", false);
//    diffuse.setSpecular(Color(1.0f,1.0f,1.0f));
    diffuse.setTransparency(0.0f);
    auto floor = Box(Position{-100,-20,-100},Position{100,-21,100}, diffuse);
    renderer.add(floor);


      // Glass Sphere
      auto transparentMaterial = Material("../scene/materials/MetalCastIron2/color.jpg", nullptr, nullptr);
      transparentMaterial.setTransparency(1.0f);
      auto glassSphere = Sphere(4, Position{-3, -3, 3}, transparentMaterial);
      renderer.add(glassSphere);

      // Red Diffuse Sphere
      auto sphereDiffuseRed = Sphere(10, Position{0, 0, -20}, diffuse);
      renderer.add(sphereDiffuseRed);

      // Rusted iron - combined
      auto rustedIronMat = Material("../scene/materials/MetalCastIron2/color.jpg",
                                    "../scene/materials/MetalCastIron2/matalness.jpg",
                                    "../scene/materials/MetalCastIron2/roughness.jpg");
      rustedIronMat.setSpecular({1.0f, 1.0f, 1.0f});
      auto rustedIronSphere = Sphere(3, Position{5, 5, 5}, rustedIronMat);
      renderer.add(rustedIronSphere);


    // Standford Bunny
//    auto bunny = MeshObject(loadObjFile("sphere.obj"), true);
//    renderer.add(bunny);

    // Pass the texture to the program as uniform input called "Texture"
    program.setUniform("Texture", framebuffer);

    // Set Matrices to identity so there are no projections/transformations applied in the vertex shader
    program.setUniform("ModelMatrix", mat4{});
    program.setUniform("ViewMatrix", mat4{});
    program.setUniform("ProjectionMatrix", mat4{});
  }

  /*!
   * Render window content when needed
   */
  void onIdle() override {
    // Get the start time of the execution
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    renderer.render(10);

    // Get the end time of the execution and log the duration into the stdout
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = chrono::duration_cast<chrono::duration<double>> (end - start);
    cout << "Rendering time: " << duration.count() << "\n";

    auto& samples = renderer.samples;

    // Generate the framebuffer
    auto& image = framebuffer.image;
    #pragma omp parallel for
    for (int y = 0; y < image.height; ++y) {
      for (int x = 0; x < image.width; ++x) {
        Color& color = samples[image.width*y+x].color;

        Color visible;
        visible.r = color.r;
        visible.g = color.g;
        visible.b = color.b;

//        // Exposure
//        visible.r *= 16.0f;
//        visible.g *= 16.0f;
//        visible.b *= 16.0f;

        // Clamp
        color = clamp(color, 0.0f, 1.0f);

        if (TONE_MAPPING == ToneMapping::GLOBAL_REINHARD) {
          float luminance = 0.212671f * color.r + 0.71516f * color.g + 0.072169f * color.b;
          float luminanceD = luminance / (1.0f + luminance);

          visible.r = luminanceD * visible.r / luminance;
          visible.g = luminanceD * visible.g / luminance;
          visible.b = luminanceD * visible.b / luminance;

        } else if (TONE_MAPPING == ToneMapping::LINEAR) {
          ;
        } else if (TONE_MAPPING == ToneMapping::UNCHARTED_2) {
          visible = Uncharted2Tonemap(ExposureBias * visible);

          Color whiteScale = 1.0f/Uncharted2Tonemap({W, W, W});
          visible *= whiteScale;
        }

        // Gamma Correction
        float powVal = 1/2.2f;
        visible.r = pow(visible.r, powVal);
        visible.g = pow(visible.g, powVal);
        visible.b = pow(visible.b, powVal);

        image.setPixel(x, y, (float) visible.r, (float) visible.g, (float) visible.b);
      }
    }
    framebuffer.update();

    // Set gray background
    glClearColor(.5f, .5f, .5f, 0);

    // Clear depth and color buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render the quad geometry
    quad.render();
  }
};

int main() {
  // Create a window with OpenGL 3.3 enabled
  PathTracerWindow window;

  // Initialize Tone Mapping
  TONE_MAPPING = ToneMapping::UNCHARTED_2;

  // Main execution loop
  while (window.pollEvents()) {}

  return EXIT_SUCCESS;
}
