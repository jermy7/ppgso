#pragma once

#include "shape.h"

/*!
 * Structure representing a sphere which is defined by its center position, radius and material
 */
struct Sphere final : public Shape {
public:
    Distance radius;
    Position center;
    const Material material;

    Sphere(Distance r, const Position &c, const Material &m) : radius{r}, center{c}, material{m} {}

    inline Hit intersect(const Ray &ray) const override {
        auto oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float dis = b * b - a * c;

        if (dis > 0) {
            float e = sqrtf(dis);
            float t = (-b - e) / a;

            if (t > EPS) {
                auto pt = ray.point(t);
                auto n = normalize(pt - center);
                return {t, pt, n, &material};
            }

            t = (-b + e) / a;

            if (t > EPS) {
                auto pt = ray.point(t);
                auto n = normalize(pt - center);
                return {t, pt, n, &material};
            }
        }
        return noHit;
    }
};