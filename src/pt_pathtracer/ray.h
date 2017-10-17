#pragma once

#include <glm/glm.hpp>

// Global constants
constexpr float INF = std::numeric_limits<float>::max();       // Will be used for infinity
constexpr float EPS = std::numeric_limits<float>::epsilon();   // Numerical epsilon
const float DELTA = std::sqrt(EPS);                             // Delta to use

// Definitions of common types for readability
using Vector = glm::vec3;
using Direction = Vector;
using Position = Vector;
using Distance = float;
using Color = Vector;
using TexCoord = glm::vec2;

struct Ray {
  Position origin;
  Direction direction;

  inline Position point(Distance t) const {
    return origin + direction * t;
  }
};

/*!
 * Generate a normalized vector that sits on the surface of a half-sphere which is defined using a normal. Used to generate random diffuse reflections.
 * @param normal Normal that defines the dome/half-sphere direction
 * @return Random 3D vector on the dome surface
 */
inline Direction RandomDome(const Direction &normal) {
  double d;
  Direction p;

  do {
    p = glm::sphericalRand(1.0);
    d = dot(p, normal);
  } while(d < 0);

  return p;
}

inline Direction CosineSampleHemisphere(const Direction &normal)
{
  static const double SQRT_OF_ONE_THIRD = sqrt(1/3.0);
  float up = sqrt(glm::linearRand(0.0,1.0)); // cos(theta)
  float over = sqrt(1 - up * up); // sin(theta)
  float around = glm::linearRand(0.0,2.0) * M_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

  Direction directionNotNormal;
  if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = {1, 0, 0};
  } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    directionNotNormal = {0, 1, 0};
  } else {
    directionNotNormal = {0, 0, 1};
  }

  // Use not-normal direction to generate two perpendicular directions
  Direction perpendicularDirection =
      glm::normalize(glm::cross(normal, directionNotNormal));

  return up * normal
         + std::cos(around) * over * perpendicularDirection
         + std::sin(around) * over * (glm::normalize(glm::cross(normal, perpendicularDirection)));
}
