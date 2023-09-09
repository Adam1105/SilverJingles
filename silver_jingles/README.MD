# Silver Jingles: A Neural Vocoder/Codec Library that is aimed to last!

## Overview
Silver Jingles is a powerful audio generation library designed specifically for neural vocoders and codecs. It offers a straightforward and flexible approach to audio generation without the complexity of configuration files commonly found in other libraries. The library trust in the importance of explicit code over cumbersome configuration files, as it simplifies components re-usability across various frameworks.

## Key Principles
The library is built on solid software development principles, ensuring a clean and maintainable codebase:

1. **Single Responsibility**: We adhere to the Single Responsibility Principle, separating various aspects of the library into dedicated components. The core model focuses on the architecture and forward method, while other responsibilities like computing loss, validation steps, and training/testing recipes are handled by specialized classes or within the frameworks that utilize the library.

2. **Open for Extension, Closed for Modification**: Unlike some frameworks that require modifying existing classes to add functionality, our library is designed to allow easy extension without altering existing code. This encourages the reusability of existing components and demonstrates how to build on them by creating new classes.

3. **Liskov Substitution**: We provide a modular and inheritance-friendly structure that facilitates the implementation of new models. Whether you're working with BigVSAN, BigVGAN, HiFiGAN, Multi-Band-Melgan, or UnivNet, EnCodec, or HiFiNET, the library offers a coherent and adaptable framework for various model implementations.

4. **Interface Segregation**: While the library addresses the needs of different models, it avoids the bloat often seen in full-framework-like repositories. We prioritize the simplicity and reusability of individual components without unnecessary complexity.

5. **Dependency Inversion**: **!A central tenet of the library!** is dependency inversion. Modules within the library do not initialize objects within their constructors. Instead, they expect already created Modules for initialization, promoting modularity and flexibility. Specific models are responsible for initializing these layers as needed.

## Reusability and Recipes
The library emphasizes reusability, making it easy to integrate into your projects and workflows. Whether you need pre-trained neural vocoder/codec models or modular components, Silver Jingles provides a one-stop-shop solution.

## Get Involved
To truly benefit from this library's potential, we encourage you to contribute. Instead of creating new packages for every model, consider importing discriminators, losses, and activations from Silver Jingles. Together, we can build a stronger and more unified audio generation community!
