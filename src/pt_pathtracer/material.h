#pragma once

#include "ray.h"

enum class MaterialType {
  DIFFUSE,
  SPECULAR,
  REFRACTIVE
};

/*!
 * Material coefficients for diffuse and emission
 */
struct Material {
  Color emission, diffuse;
  MaterialType type = MaterialType::DIFFUSE;

  static const Material Light() {
      return {{1, 1, 1}};
  };
  static const Material Red() {
      return {{}, {1, 0, 0}};
  };
  static const Material Green() {
      return {{}, {0, 1, 0}};
  };
  static const Material Blue() {
      return {{}, {0, 0, 1}};
  };
  static const Material Yellow() {
      return {{}, {1, 1, 0}};
  };
  static const Material Magenta() {
      return {{}, {1, 0, 1}};
  };
  static const Material Cyan() {
      return {{}, {0, 1, 1}};
  };
  static const Material White() {
    return {{}, {1, 1, 1}};
  };
  static const Material Gray() {
    return {{}, {.5, .5, .5}};
  };
  static const Material Mirror() {
    return {{}, {}, MaterialType ::SPECULAR};
  };
  static const Material Glass() {
    return {{}, {}, MaterialType::REFRACTIVE};
  };
};

