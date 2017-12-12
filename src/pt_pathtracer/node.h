#pragma once

#include "shape.h"
#include "bounding_box.h"

using Axis = short; // 0: X, 1: Y, 2: Z

const int BOXES_IN_LEAF = 20;

struct Node {
  Axis axis;
  float midpoint;
  std::shared_ptr<BoundingBox> box; // Bounding box for this node - for intersections
  std::vector<std::shared_ptr<BoundingBox>> boxes; // Bounding boxes of triangles
  std::shared_ptr<Node> left, right;

  // Node initialization
  Node(Axis axis, std::vector<std::shared_ptr<BoundingBox>> boxes, int depth) {
//    std::cout << "Depth: " << depth << ", Axis: " << axis << ", Boxes count: " << boxes.size() << "\n";
    this->axis = axis;
    this->boxes = boxes;

    // Find bounds for this node's box
    Position min = {INF, INF, INF};
    Position max = {-INF, -INF, -INF};
    for (auto box : boxes) {
      for (int i = 0; i < 3; i++) {
        min[i] = (min[i] > box->min[i]) ? box->min[i] : min[i];
        max[i] = (max[i] < box->max[i]) ? box->max[i] : max[i];
      }
    }
    this->box = std::make_shared<BoundingBox>(min, max);

    //Should I split?
    if (this->boxes.size() > BOXES_IN_LEAF) {
      // Calculate a midpoint - for now using arithmetic middle, can replace with median
      this->midpoint = (max[axis] + min[axis]) / 2.0f;

      // Separate boxes based on midpoint
      std::vector<std::shared_ptr<BoundingBox>> leftBoxes;
      std::vector<std::shared_ptr<BoundingBox>> rightBoxes;
      for (auto box: boxes) {
        if (box->min[axis] < midpoint) {
          leftBoxes.emplace_back(box);
        } else {
          rightBoxes.emplace_back(box);
        }
      }

      // Initialize sub-nodes
      float new_axis = (axis + 1) % 3;
      this->left = std::make_shared<Node>(new_axis, leftBoxes, depth + 1);
      this->right = std::make_shared<Node>(new_axis, rightBoxes, depth + 1);
    }
  }
};
