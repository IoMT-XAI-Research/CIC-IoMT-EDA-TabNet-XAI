# Docker Instructions

This project has been Dockerized to ensure portability and ease of use. You can run the entire system (Backend + Traffic Simulator) using Docker Compose.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop).

## How to Run

1.  **Open a terminal** in the root directory of the project.

2.  **Build and Start the containers**:
    ```bash
    docker-compose up --build
    ```
    This command will:
    - Build the Python image with all dependencies.
    - Start the `backend` service on port `8000`.
    - Start the `simulator` service, which will begin replaying traffic and sending alerts to the backend.

3.  **Access the Backend**:
    - The API is available at `http://localhost:8000`.
    - API Documentation: `http://localhost:8000/docs`.

4.  **View Logs**:
    - You will see logs from both services in your terminal.
    - Watch for "🚨 ATTACK DETECTED!" messages from the simulator and "Alert Sent" confirmations.

5.  **Stop the System**:
    - Press `Ctrl+C` in the terminal to stop the containers.
    - To remove the containers, run:
        ```bash
        docker-compose down
        ```

## Troubleshooting

- **Port Conflicts**: If port `8000` is already in use, modify the `ports` mapping in `docker-compose.yml` (e.g., `"8001:8000"`).
- **Missing Data**: Ensure the `data/` and `artifacts/` directories contain the necessary files (`.csv` files for simulation, `.zip` and `.pkl` files for the model).
