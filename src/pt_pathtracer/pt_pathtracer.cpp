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

#include "renderer.h"
#include "sphere.h"
#include "box.h"

using namespace std;
using namespace glm;
using namespace ppgso;

const unsigned int SIZE = 512;

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
public:
  /*!
   * Construct a new Window and initialize shader uniform variables
   */
  PathTracerWindow() : Window{"pt_pathtracer", SIZE, SIZE}, renderer{SIZE, SIZE}, framebuffer{SIZE, SIZE} {
    // Prepare the scene
    auto& scene = renderer.scene;

    renderer.camera.position = {0,0,15};

    // Boxes
    auto floor = Box(Position{-10,-11,-10},Position{10,-10,20},Material::White());
    renderer.add(floor);
    auto leftWall = Box(Position{-11,-10,-10},Position{-10,10,20},Material::Red());
    renderer.add(leftWall);
    auto rightWall = Box(Position{10,-10,-10},Position{11,10,20},Material::Green());
    renderer.add(rightWall);
    auto backWall = Box(Position{-10,-10,-11},Position{10,10,-10},Material::Gray());
    renderer.add(backWall);
    auto frontWall = Box(Position{-10,-10,20},Position{10,10,21},Material::Gray());
    renderer.add(frontWall);
    auto ceiling = Box(Position{-10,10,-10},Position{10,11,20},Material::Light());
    renderer.add(ceiling);

    // Box with rotation
    auto yellowBox = Box(Position{-1,0,-1},Position{1,1,1},Material::Yellow());
    auto transformedBox = TransformedShape(yellowBox);
    transformedBox.position = {0,-10,0};
    transformedBox.rotation = {0,0,M_PI/3.0};
    transformedBox.scale = {2,5,2};
    renderer.add(std::move(transformedBox));

     // Spheres
    auto glassSphere = Sphere(2,Position{-5,-7,3},Material::Glass());
    renderer.add(glassSphere);
    auto cornerSphere = Sphere(10,Position{ 10,10,-10},Material::Blue());
    renderer.add(cornerSphere);

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
    renderer.render();
    auto& samples = renderer.samples;

    // Generate the framebuffer
    #pragma omp parallel for
    auto& image = framebuffer.image;
    for (int y = 0; y < image.height; ++y) {
      for (int x = 0; x < image.width; ++x) {
        Color& color = samples[image.width*y+x].color;
        image.setPixel(x, y, (float) color.r, (float) color.g, (float) color.b);
      }
    }
    framebuffer.update();

    // Set gray background
    glClearColor(.5f, .5f, .5f, 0);

    // Clear depth and color buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render the quad geometry
    quad.render();

    // Update the window contents
    swap();
  }
};

int main() {
  // Create a window with OpenGL 3.3 enabled
  PathTracerWindow window;

  // Main execution loop
  while (window.pollEvents()) {}

  return EXIT_SUCCESS;
}
