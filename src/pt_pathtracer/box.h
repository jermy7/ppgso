#pragma once

#include "shape.h"

/*!
 * Structure representing a sphere which is defined by its center position, radius and material
 */
struct Box final : public Shape {
  Position min, max;
  const Material material;

  Box(const Position& mi, const Position& ma, const Material& m) : min{mi}, max{ma}, material{m} {}

  inline virtual Hit intersect(const Ray &ray) const override {
    Vector n1 = (min - ray.origin) / ray.direction;
    Vector f1 = (max - ray.origin) / ray.direction;
    auto n = glm::min(n1,f1);
    auto f = glm::max(n1,f1);
    float t0 = std::max(std::max(n.x, n.y), n.z);
    float t1 = std::min(std::min(f.x, f.y), f.z);

    Vector point = ray.point(t0);
    Vector normal;
    if (point.x < min.x + EPS)
        normal = {-1, 0, 0};
    else if (point.x > max.x - EPS)
        normal = {1, 0, 0};
    else if (point.y < min.y + EPS)
        normal = {0, -1, 0};
    else if (point.y > max.y - EPS)
        normal = {0, 1, 0};
    else if (point.z < min.z + EPS)
        normal = {0, 0, -1};
    else if (point.z > max.z - EPS)
        normal = {0, 0, 1};
    else normal = {0, 1, 0};

    if (t0 > 0.0 && t0 < t1) {
      return Hit{t0, point, normal, &material};
    }
    return noHit;
  }
};