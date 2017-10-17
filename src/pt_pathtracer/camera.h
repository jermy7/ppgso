#include "ray.h"

/*!
 * Structure representing a simple camera that is composed on position, up, back and right vectors
 */
struct Camera {
  Position position = {0,0,0};
  Direction up = {0,1,0}, direction = {0,0,-1}, right = {1,0,0};

  // Distance of the projection plane from the camera position
  Distance D = 1;

  /*!
   * Generate a new Ray for the given viewport size and position
   * @param x Horizontal position in the viewport
   * @param y Vertical position in the viewport
   * @param width Width of the viewport
   * @param height Height of the viewport
   * @return Ray for the giver viewport position with small random deviation applied to support multi-sampling
   */
  inline Ray generateRay(int x, int y, int width, int height) const {
    // Aspect ration
    float ratio = (float)width/(float)height;
    // Camera deltas
    auto vdu = 2.0f * right / (float)width;
    auto vdv = 2.0f * ratio * -up / (float)height;

    Ray ray;
    ray.origin = position;
    ray.direction = direction * D
                    + vdu * ((float)(-width/2 + x) + glm::linearRand(0.0f, 1.0f))
                    + vdv * ((float)(-height/2 + y) + glm::linearRand(0.0f, 1.0f));
    ray.direction = normalize(ray.direction);
    return ray;
  }
};