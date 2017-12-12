#pragma once

#include "ray.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

/*!
 * Material coefficients for diffuse and emission
 */
struct Material {
private:
    int width, height;
    std::vector<Color> albedo;
    std::vector<float> metalness;
    std::vector<float> roughness;
    float transparency = 0.0f;
    Color specularColor = {0.21f,0.21f,0.21f};
    float ior = 1.5f;

    bool isLight = false;
    Color lightColor;


    void positionToCoord(Position p, int &x, int &y) const {
        // Hack: Since the camera is static, X and Y coordinates of the normal are enough to get the color from the sphere
        // NOTE: This works for the sphere!
        x = (int)(((p.x + 1.0f) / 2.0f) * width);
        y = (int)(((-p.y + 1.0f) / 2.0f) * height); // Convert to 0 - MAX, instead of MAX - 0
    }

public:
    // Color texture, i.e. albedo
    void setTexture(char *texture, std::vector<Color> *vec) {
        int n;

        stbi_hdr_to_ldr_gamma(2.2f);
        stbi_hdr_to_ldr_scale(1.0f);
        unsigned char *data = stbi_load(texture, &width, &height, &n, 3);
        if (data != nullptr) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    vec->push_back({
                                           data[(i * width + j) * n + 0] / 255.0f,
                                           data[(i * width + j) * n + 1] / 255.0f,
                                           data[(i * width + j) * n + 2] / 255.0f
                                   });
                }
            }
        }
        stbi_image_free(data);
    }

    // float texture, i.e. roughness
    void setTexture(char *texture, std::vector<float> *vec, bool invert=false) {
        int n;

        stbi_hdr_to_ldr_gamma(2.2f);
        stbi_hdr_to_ldr_scale(1.0f);
        unsigned char *data = stbi_load(texture, &width, &height, &n, 1);
        if (data != nullptr) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    float value;
                    value = data[i * width + j] / 255.0f;

                    // i.e. inverting glosiness map to roughness
                    if (invert) {
                        value = 1 - value;
                    }
                    vec->push_back(value);
                }
            }
        }
        stbi_image_free(data);
    }

    /**
     * Get albedo for given coordinates.
     * @param x
     * @param y
     * @return
     */
    Color getAlbedo(Position p) const {
        if (isLight) {
            return lightColor;
        }
        if (albedo.size() == 0) {
            return {0.0f, 0.0f, 0.0f};
        }
        int x, y;
        positionToCoord(p, x, y);
        return albedo[width * y + x];
    }

    /**
     * Get metalness for given coordinates.
     * @param x
     * @param y
     * @return
     */
    float getMetalness(Position p) const {
        if (metalness.size() == 0) {
            return 0.0f;
        }

        int x, y;
        positionToCoord(p, x, y);
        return metalness[width * y + x];
    }

    /**
     * Get roughness for given coordinates.
     * @param x
     * @param y
     * @return
     */
    float getRoughness(Position p) const {
        int x, y;
        positionToCoord(p, x, y);
        return roughness[width * y + x];
    }

    void setTransparency(float transparency) {
        this->transparency = transparency;
    }
    float getTransparency() const {
        return transparency;
    }

    void setSpecular(Color specularColor) {
        this->specularColor = specularColor;
    }

    Color getSpecular() const {
        return this->specularColor;
    }

    void setIor(float ior) {
        this->ior = ior;
    }
    /**
     * Get index of refraction for this material.
     * @return Index of refraction.
     */
    float getIor() const {
        return ior;
    }

    /**
     * Check if this material is a light source
     * @return
     */
    bool getIsLight() const {
        return this->isLight;
    }

    /**
     * Get light color
     */
    Color getLight() const {
        return this->lightColor;
    }

    Material() {};

    Material(char *albedoMap, char *metalnessMap, char *roughnessMap, bool glosiness=false) {
        setTexture(albedoMap, &albedo);
        setTexture(metalnessMap, &metalness);
        setTexture(roughnessMap, &roughness, glosiness);
    }

    Material(Color color) {
        this->isLight = true;
        this->lightColor = color;
    }
};
