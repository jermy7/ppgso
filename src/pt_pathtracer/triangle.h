#pragma once

#include "shape.h"

/*!
 * Triangle structure for meshes.
 */
struct Triangle final: public Shape {
  Vector v1, v2, v3;
  Material material;

  Triangle(Vector a, Vector b, Vector c, Material material) {
    this->v1 = a;
    this->v2 = b;
    this->v3 = c;
    this->material = material;
  }

  inline virtual Hit intersect(const Ray &ray) const override {
    Vector edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    h = glm::cross(ray.direction, edge2);
    a = glm::dot(edge1, h);
    if (a > -EPS && a < EPS)
      return noHit;
    f = 1/a;
    s = ray.origin - v1;
    u = f * (glm::dot(s, h));
    if (u < 0.0 || u > 1.0)
      return noHit;
    q = glm::cross(s, edge1);
    v = f * glm::dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0)
      return noHit;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * glm::dot(edge2, q);
    if (t > EPS) // ray intersection
    {
      return {t, ray.point(t), normalize(cross(edge1, edge2)), &material};
    }
    else // This means that there is a line intersection but not a ray intersection.
      return noHit;
  }
};