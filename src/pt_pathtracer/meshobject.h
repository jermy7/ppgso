#pragma once

#include "node.h"
#include "triangle.h"
#include "shape.h"

/*!
 * Mesh structure for meshes.
 */
struct MeshObject final: public Shape {
private:
  // KD Tree
  std::shared_ptr<Node> root;
  bool use_kd;
public:
  std::vector<Triangle> triangles;

  MeshObject(std::vector<Triangle> triangles, bool use_kd = true) {
    this->triangles = triangles;
    this->use_kd = use_kd;

    std::vector<std::shared_ptr<BoundingBox>> boxes;
    for (auto triangle: triangles) {
      auto box = std::make_shared<BoundingBox>(triangle);
      boxes.emplace_back(box);
    }

    if (this->use_kd) {
      // Initialize kd-tree with X-axis
      this->root = std::make_shared<Node>(0, boxes, 0);
    }
  }

  /**
   * Check the leaf node's bounding boxes and their triangles for intersection
   * @param boxes placed in the leaf node of ld-tree
   * @return Hit or noHit
   */
  Hit intersect_node_triangles(const Ray &ray, const std::vector<std::shared_ptr<BoundingBox>> boxes) const {
    Hit hit = noHit;
    for (auto box: boxes) {
      auto boxHit = box->triangle->intersect(ray);
      hit = (hit.distance < boxHit.distance) ? hit : boxHit;
    }
    return hit;
  }

  /**
   * Find intersection in the kd-tree.
   * @param ray cast from the camera
   * @param node in the kd-tree
   * @return Hit or noHit if the whole mesh has been missed
   */
  Hit kd_intersect(const Ray &ray, const std::shared_ptr<Node> node) const {
    // Has the node's BoundingBox been hit?
    auto hit = node->box->intersect(ray);

    // No, return hit (which is noHit)
    if (hit.distance == INF) {
      return hit;
    }

    // Yes, let's check the children nodes
    Hit leftChildHit;
    if (node->left != nullptr) {
      leftChildHit = kd_intersect(ray, node->left);
    } else {
      leftChildHit = intersect_node_triangles(ray, node->boxes);
    }

    Hit rightChildHit;
    if (node->right != nullptr) {
      rightChildHit = kd_intersect(ray, node->right);
    } else {
      rightChildHit = intersect_node_triangles(ray, node->boxes);
    }

    // Return the closer hit - one may be infinite (noHit) but they both may be a real hit
    return (leftChildHit.distance < rightChildHit.distance) ? leftChildHit : rightChildHit;
  }

  inline virtual Hit intersect(const Ray &ray) const override {
    if (this->use_kd) {
      // Will search the kd-tree for a hit
      return kd_intersect(ray, this->root);
    } else {
      Hit hit = noHit;
      for (auto triangle : triangles) {
        Hit lh = triangle.intersect(ray);
        if (lh.distance < hit.distance) {
          hit = lh;
        }
      }

      return hit;
    }
  }
};