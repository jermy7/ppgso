#pragma once

#include "triangle.h"
#include "shape.h"

struct BoundingBox final: Shape {
  Position min, max;
  std::shared_ptr<Triangle> triangle;

  // For Node
  BoundingBox(Position mi, Position ma): min{mi}, max{ma} {}

  // For leaf
  BoundingBox(Triangle triangle) {
    this->triangle = std::make_shared<Triangle>(triangle);

    this->min = {
            std::min(std::min(this->triangle->v1.x, this->triangle->v2.x), this->triangle->v3.x),
            std::min(std::min(this->triangle->v1.y, this->triangle->v2.y), this->triangle->v3.y),
            std::min(std::min(this->triangle->v1.z, this->triangle->v2.z), this->triangle->v3.z)
    };
    this->max = {
            std::max(std::max(this->triangle->v1.x, this->triangle->v2.x), this->triangle->v3.x),
            std::max(std::max(this->triangle->v1.y, this->triangle->v2.y), this->triangle->v3.y),
            std::max(std::max(this->triangle->v1.z, this->triangle->v2.z), this->triangle->v3.z)
    };
  }

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
      return Hit{t0, point, normal, nullptr};
    }
    return noHit;
  }
};
