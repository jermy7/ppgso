#pragma once

#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include "hit.h"
#include "material.h"

// Shape interface
struct Shape {
  virtual ~Shape() = default;
  virtual Hit intersect(const Ray& ray) const = 0;
};

// Shape with model matrix that can transform any other shape
struct TransformedShape final : public Shape {
  std::unique_ptr<Shape> shape;
  Vector rotation = {0, 0, 0};
  Vector scale = {1, 1, 1};
  Position position = {0, 0, 0};

  template<typename T>
  TransformedShape(T s) : shape{std::make_unique<T>(std::move(s))} {}

  virtual Hit intersect(const Ray& ray) const override {
    // Compute model matrix and inverse
    glm::mat4 matrix = glm::translate(glm::mat4(1.0f), position)
                     * glm::orientate4(rotation)
                     * glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 inverse = glm::inverse(matrix);

    // Transform ray to object space
    Ray transformedRay = { inverse * glm::vec4{ray.origin, 1.0f}, inverse * glm::vec4{ray.direction, 0.0f} };

    // Hit in object space
    auto hit = shape->intersect(transformedRay);

    // Transform to world space
    hit.position = matrix * glm::vec4{hit.position, 1.0f};
    hit.normal = glm::normalize(matrix * glm::vec4{hit.normal, 0.0f});

    return hit;
  }

};