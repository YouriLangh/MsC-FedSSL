
#utils 

The **student-teacher approach** is a learning method where a **teacher model** (a big, smart AI) trains a **student model** (a smaller, simpler AI). The teacher already knows a lot and helps the student learn by giving it **simplified or processed knowledge** instead of raw data. This makes the student learn faster and more efficiently. The goal is for the student to become almost as good as the teacher while being **smaller, faster, and easier to use**.

### How It Relates to Federated Learning

Federated learning is a method where many devices (like phones or computers) **train a model together** without sharing their private data. Instead of sending data to a central server, each device trains its own small model and **only sends updates (not raw data) to improve a global model**.

The **student-teacher approach** can help here by making sure the global model (teacher) can **train smaller models (students) on each device**. This way:

- The student models learn faster because they get help from the teacher.
- The student models can run efficiently on smaller devices.
- The system stays privacy-friendly because devices don’t need to send raw data.