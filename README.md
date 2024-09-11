# End-to-End Machine Learning Project

## Project Overview
This project demonstrates a complete pipeline for building, deploying, and maintaining a machine learning model in production.


## Prerequisites
- **Python 3.9+**
- **Virtual Environment**: Create a virtual environment using `venv` or `conda`


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AbdullahMansoor123/ml_end2end
    ```
2. Navigate to the project directory:
    ```bash
    cd end-to-end-ml-project
    ```
3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run Flask App**: 
    - Run the app by running `python app.py` in the terminal with virtual environment enabled.


# CICD with AWS

### Steps

1. **Clone the GitHub Repo**  
   ```bash
   git clone https://github.com/AbdullahMansoor123/ml_end2end
   cd ml_end2end
   ```

2. **Create a Dockerfile**  
   - Define the Dockerfile to build the application.

3. **Create a GitHub Workflow**  
   - Set up GitHub Actions workflow for continuous integration.

4. **Create IAM User and Download Access Key**  
   - Create an IAM user in AWS with necessary permissions and download the access key and secret key.

5. **Create AWS ECR Repo and Copy URL**  
   - Create a repository in AWS Elastic Container Registry (ECR) and copy the repository URL.

6. **Set Up EC2 Instance**  
   - Launch an EC2 instance and configure it as follows:
     ```bash
     sudo apt-get update -y
     sudo apt-get update
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     sudo usermod -aG docker ubuntu
     newgrp docker
     ```
     check if the setup completed successfully by running `docker images` on the EC2 instance

7. **Create Runner on GitHub to Connect Repo with EC2**
   - **Create a Folder for the Runner**  
     ```bash
     mkdir actions-runner && cd actions-runner
     ```

   - **Download the Latest Runner Package**  
     ```bash
     curl -o actions-runner-linux-x64-2.319.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
     ```

   - **Optional: Validate the Hash**  
     ```bash
     echo "3f6efb7488a183e291fc2c62876e14c9ee732864173734facc85a1bfb1744464  actions-runner-linux-x64-2.319.1.tar.gz" | shasum -a 256 -c
     ```

   - **Extract the Installer**  
     ```bash
     tar xzf ./actions-runner-linux-x64-2.319.1.tar.gz
     ```

   - **Configure the Runner**  
     ```bash
     ./config.sh --url https://github.com/AbdullahMansoor123/ml_end2end --token <your_github_token>
     ```

   - **Run the Runner**  
     ```bash
     ./run.sh
     ```

   - **Update GitHub Workflow YAML**  
     Use the following in your GitHub Actions workflow file:
     ```yaml
     runs-on: self-hosted
     ```

8. **Add GitHub Secrets for AWS**
   - Add AWS credentials (access key and secret key) as GitHub secrets for your workflow to access AWS services.

9. **Edit Inbound Rules for EC2 Instance**  
   - Go to the **Security** settings of your EC2 instance.
   - Navigate to **Security Groups**.
   - Edit the **Inbound Rules** by adding a **Custom TCP** rule with the port number `4242` as specified in the GitHub workflow file
10.  **Initiate CICD github workflow**
   - Commit and push any change in the git repo to intiate cicd workflow.  