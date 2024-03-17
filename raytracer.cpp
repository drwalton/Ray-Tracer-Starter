﻿// A practical implementation of the ray tracing algorithm.

#include "geometry.h"
#include "SDL.h" 
#include <fstream>
#include <chrono>

#define M_PI 3.14159265359

SDL_Window* window;
SDL_Renderer* renderer;
SDL_Surface* screen;

void init() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow(
        "Software Ray Tracer",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        640,
        480,
        0
    );

    screen = SDL_GetWindowSurface(window);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
}

void putpixel(SDL_Surface* surface, int x, int y, Uint32 pixel)
{
    int bpp = surface->format->BytesPerPixel;
    /* Here p is the address to the pixel we want to set */
    Uint8* p = (Uint8*)surface->pixels + y * surface->pitch + x * bpp;

    switch (bpp) {
    case 1:
        *p = pixel;
        break;

    case 2:
        *(Uint16*)p = pixel;
        break;

    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        }
        else {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;

    case 4:
        *(Uint32*)p = pixel;
        break;
    }
}

struct Ray {
    Point3f o;
    Vec3f d;
    Ray(const Point3f& o, const Vec3f& d) : o(o), d(d) {}
};

struct Sphere {
    Point3f c;
    double r;
    Sphere(const Point3f& c, double r) : c(c), r(r) {}
    Vec3f getNormal(const Vec3f& pi) const { return (pi - c) / r; }

    // Solve t^2*d.d+2*t*(o-p).d+(o-p).(o-p)-R^2=0​
    bool intersect(const Ray& ray, double& t) const {
        // ****** YOUR WORK HERE ******
        // 
        // Implement your ray-sphere intersection here!
        // This should find out whether intersection occurs at all - if not it should return false.
        // If intersection does occur, it should return true, after setting "t" to the parameter 
        // of the ray.

        //Steps:
        // First find the discriminant of the quadratic equation (b^2 - 4ac)
        // If this is less than zero, then the equation has no solutions, and the ray
        // misses the sphere. Return False.
        // Otherwise, complete solving the quardratic equation as usual. 
        // This gives two solutions for t.
        // If both are negative, the intersection doesn't count as it's behind the start of the ray.
        // Return false.
        // Otherwise, set t the smallest non-negative root and return True.

        // Right now the code just always says "No Intersection" - replace this with working code!
        return false;
    }
};

// method to ensure colours don’t go out of 8 bit range in RGB​
void clamp255(Vec3f& col) {
    col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
    col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
    col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main(int argc, char **argv)
{
    // initialise SDL2
    init();

    const Colour white(255, 255, 255);
    const Colour black(0, 0, 0);
    const Colour red(255, 0, 0);

    const Sphere sphere(Vec3f(screen->w * 0.5, screen->h * 0.5, 50), 50);
    const Sphere light(Vec3f(0, 0, 50), 1);

    double t;
    Colour pix_col(black);

    SDL_Event e;
    bool running = true;
    while (running) {

        auto t_start = std::chrono::high_resolution_clock::now();

        // clear back buffer, pixel data on surface and depth buffer (as movement)
        SDL_FillRect(screen, nullptr, SDL_MapRGB(screen->format, 0, 0, 0));
        SDL_RenderClear(renderer);

        for (int y = 0; y < screen->h; ++y) {
            for (int x = 0; x < screen->w; ++x) {
                pix_col = black;
                const Ray ray(Point3f(x, y, 0), Vec3f(0, 0, 1));
                if (sphere.intersect(ray, t)) {
                    const Vec3f pi = ray.o + ray.d * t;
                    Vec3f L = light.c - pi;
                    Vec3f N = sphere.getNormal(pi);
                    Vec3f Ln = L.normalize();
                    double dt = Ln.dotProduct(N.normalize());
                    pix_col = (red + white * dt) * 0.5;
                    clamp255(pix_col);
                }
                Uint32 colour = SDL_MapRGB(screen->format, pix_col.x, pix_col.y, pix_col.z);
                putpixel(screen, x, y, colour);
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        auto passedTime = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cerr << "Frame render time:  " << passedTime << " ms" << std::endl;

        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, screen);
        if (texture == NULL) {
            fprintf(stderr, "CreateTextureFromSurface failed: %s\n", SDL_GetError());
            exit(1);
        }
        SDL_FreeSurface(screen);

        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        SDL_DestroyTexture(texture);

        if (SDL_PollEvent(&e))
        {
            switch (e.type) {
            case SDL_KEYDOWN:
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = false;
                    break;
                }
                break;
            }
        }
    }

    return 0;
}
