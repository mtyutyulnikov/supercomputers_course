#include "minirt/minirt.h"
//#include "C://MINIRT//include//minirt//minirt.h"
#include <iostream>
#include <chrono>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>

using namespace minirt;

void initScene(Scene &scene) {
    Color red{1, 0.2, 0.2};
    Color blue{0.2, 0.2, 1};
    Color green{0.2, 1, 0.2};
    Color white{0.8, 0.8, 0.8};
    Color yellow{1, 1, 0.2};

    Material metallicRed{red, white, 50};
    Material mirrorBlack{Color{0.0}, Color{0.9}, 1000};
    Material matteWhite{Color{0.7}, Color{0.3}, 1};
    Material metallicYellow{yellow, white, 250};
    Material greenishGreen{green, 0.5, 0.5};

    Material transparentGreen{green, 0.8, 0.2};
    transparentGreen.makeTransparent(1.0, 1.03);
    Material transparentBlue{blue, 0.4, 0.6};
    transparentBlue.makeTransparent(0.9, 0.7);

    scene.addSphere(Sphere{{0, -2, 7}, 1, transparentBlue});
    scene.addSphere(Sphere{{-3, 2, 11}, 2, metallicRed});
    scene.addSphere(Sphere{{0, 2, 8}, 1, mirrorBlack});
    scene.addSphere(Sphere{{1.5, -0.5, 7}, 1, transparentGreen});
    scene.addSphere(Sphere{{-2, -1, 6}, 0.7, metallicYellow});
    scene.addSphere(Sphere{{2.2, 0.5, 9}, 1.2, matteWhite});
    scene.addSphere(Sphere{{4, -1, 10}, 0.7, metallicRed});

    scene.addSphere(Sphere{{1.5, -0.5 + 0.2, 7}, 10, transparentGreen});
    scene.addSphere(Sphere{{-2, -1 + 0.2, 6}, 5, metallicYellow});
    scene.addSphere(Sphere{{2.2, 0.5 + 0.2, 9}, 4, matteWhite});
    scene.addSphere(Sphere{{4, -1 + 0.2, 10}, 3, metallicRed});

    scene.addLight(PointLight{{-15, 0, -15}, white});
    scene.addLight(PointLight{{1, 1, 0}, blue});
    scene.addLight(PointLight{{0, -10, 6}, red});

    scene.setBackground({0.05, 0.05, 0.08});
    scene.setAmbient({0.1, 0.1, 0.1});
    scene.setRecursionLimit(20);

    scene.setCamera(Camera{{0, 0, -20},
                           {0, 0, 0}});
}

void PixelTask(const Scene &scene, Image &image, ViewPlane &viewPlane, int x, int y) {
    const auto color = viewPlane.computePixel(scene, x, y, 1);
    image.set(x, y, color);
}

struct Point {
    int x;
    int y;

    Point(int x, int y) : x(x), y(y) {}
};


template<typename T>
class ParallelQueue {
public:
    void PushBack(T value) {
        std::lock_guard<std::mutex> guard(mutexLocker);
        deque_.push_back(value);
        isNotEmpty.notify_one();
    }

    T PopFront() {
        std::unique_lock<std::mutex> lock(mutexLocker);
        while (deque_.empty())
            isNotEmpty.wait(lock);
        T front = deque_.front();
        deque_.pop_front();
        return front;
    }

    std::deque<T> deque_;
    std::mutex mutexLocker;
    std::condition_variable isNotEmpty;
};

class ThreadPool {
public:
    ParallelQueue<Point> tasks;
    std::vector<std::thread> workers;
    Scene scene;
    Image image;
    ViewPlane viewPlane;
    int threadsNum;

    ThreadPool(int threadsNum, const Scene &scene, const Image &image,
               ViewPlane &viewPlane) : viewPlane(viewPlane) {
        this->scene = scene;
        this->image = image;
        this->threadsNum = threadsNum;
        InitWorkers(threadsNum);
    };

    void InitWorkers(int threadsNum) {
        for (int i = 0; i < threadsNum; i++) {
            workers.emplace_back([this]() {
                WorkerRoutine();
            });
        }
    }

    void AddPointToTasks(Point par) {
        tasks.PushBack(par);
    }

    void WorkerRoutine() {
        auto point = tasks.PopFront();
        while (point.x >= 0) {
            PixelTask(scene, image, viewPlane, point.x, point.y);
            point = tasks.PopFront();
        }
    }

    void Join() {
        for (int i = 0; i< this->threadsNum; i++)
            tasks.PushBack({-1, -1});

        for (auto &worker: workers)
            worker.join();

        workers.clear();
    }
};

int main(int argc, char **argv) {
    int viewPlaneResolutionX = (argc > 1 ? std::stoi(argv[1]) : 2560);
    int viewPlaneResolutionY = (argc > 2 ? std::stoi(argv[2]) : 1440);
    std::string sceneFile = (argc > 4 ? argv[4] : "");

    Scene scene;
    if (sceneFile.empty()) {
        initScene(scene);
    } else {
        scene.loadFromFile(sceneFile);
    }

    const double backgroundSizeX = 4;
    const double backgroundSizeY = 4;
    const double backgroundDistance = 15;

    const double viewPlaneDistance = 5;
    const double viewPlaneSizeX = backgroundSizeX * viewPlaneDistance / backgroundDistance;
    const double viewPlaneSizeY = backgroundSizeY * viewPlaneDistance / backgroundDistance;

    ViewPlane viewPlane{viewPlaneResolutionX, viewPlaneResolutionY,
                        viewPlaneSizeX, viewPlaneSizeY, viewPlaneDistance};

    Image image(viewPlaneResolutionX, viewPlaneResolutionY); // computed image

    auto start = std::chrono::high_resolution_clock::now();

    auto threadPool = new ThreadPool(8, scene, image, viewPlane);
    for (int x = 0; x < viewPlaneResolutionX; x++) {
        for (int y = 0; y < viewPlaneResolutionY; y++) {
            threadPool->AddPointToTasks({x, y});
        }
    }
    threadPool->Join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> executionTime = end - start;
    std::cout << "Time " << executionTime.count() << std::endl;

    image.saveJPEG("raytracing.jpg");

    return 0;
}
