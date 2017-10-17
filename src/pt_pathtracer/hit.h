#pragma once

#include "ray.h"
#include "material.h"

struct Hit {
  Distance distance;
  Position position;
  Direction normal;
  const Material* material;
};

const Hit noHit{ INF, {0,0,0}, {0,0,0}, nullptr };
