import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


class AffineCorrectionAgent(nn.Module):
    def __init__(self):
        super(AffineCorrectionAgent, self).__init__()
        # Input size changed to 128*128*3 for color images
        self.fc1 = nn.Linear(128 * 128 * 3, 128)  # Hidden layer with 128 units
        self.fc2 = nn.Linear(128, 6)  # Output layer

    def forward(self, state):
        state = state.flatten()
        hidden = F.relu(self.fc1(state))  # ReLU activation for hidden layer
        affine_params = torch.tanh(self.fc2(hidden))  # Output layer with tanh activation
        # Scale parameters to reasonable ranges
        tx = affine_params[0] * 100  # Translation x
        ty = affine_params[1] * 100  # Translation y
        shear_x = affine_params[2] * 0.5  # Shear x
        shear_y = affine_params[3] * 0.5  # Shear y
        scale_x = affine_params[4] * 0.5 + 1  # Scale x (1 to 1.5)
        scale_y = affine_params[5] * 0.5 + 1  # Scale y (1 to 1.5)
        return torch.stack([tx, ty, shear_x, shear_y, scale_x, scale_y])


def calculate_similarity_reward(original_image, transformed_image):
    original_tensor = torch.tensor(original_image / 255.0, dtype=torch.float32)
    transformed_tensor = torch.tensor(
        transformed_image / 255.0, dtype=torch.float32)
    mse_loss = nn.MSELoss()
    mse = mse_loss(transformed_tensor, original_tensor)
    reward = -mse  # Reward is negative MSE, to maximize reward, minimize MSE
    return reward.item()  # Return scalar reward value


def apply_affine_transform(image, params):
    tx, ty, shear_x, shear_y, scale_x, scale_y = params
    M = np.float32([[scale_x, shear_x, tx], [shear_y, scale_y, ty]])
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness


def load_and_compare_sharpness(image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None:
        raise ValueError(f"Error: 画像 '{image_path1}' の読み込みに失敗しました。パスを確認してください。")
    if image2 is None:
        raise ValueError(f"Error: 画像 '{image_path2}' の読み込みに失敗しました。パスを確認してください。")

    sharpness1 = calculate_sharpness(image1)
    sharpness2 = calculate_sharpness(image2)

    if sharpness1 >= sharpness2:
        sharper_image = image1
        to_correct_image = image2
        sharper_image_path = image_path1
        to_correct_image_path = image_path2
        sharpness_sharper = sharpness1
        sharpness_to_correct = sharpness2
    else:
        sharper_image = image2
        to_correct_image = image1
        sharper_image_path = image_path2
        to_correct_image_path = image_path1
        sharpness_sharper = sharpness2
        sharpness_to_correct = sharpness1

    print(
        f"Sharper image: {sharper_image_path} (Sharpness: {sharpness_sharper:.2f})")
    print(
        f"Less sharp image: {to_correct_image_path} (Sharpness: {sharpness_to_correct:.2f})")

    return sharper_image, to_correct_image, sharper_image_path, to_correct_image_path


def train_affine_correction_agent(sharper_image, image_to_correct):
    original_image = sharper_image
    agent = AffineCorrectionAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.005)  # Increased learning rate to 0.005

    # Resize images once before the loop
    resized_original = cv2.resize(original_image, (128, 128))
    resized_to_correct = cv2.resize(image_to_correct, (128, 128))

    # Convert to tensors
    original_tensor = torch.tensor(resized_original / 255.0, dtype=torch.float32)

    epochs = 100
    best_reward = float('-inf')
    best_transformed_image = None

    for epoch in range(epochs):
        # Create input tensor
        state_tensor = torch.tensor(resized_to_correct / 255.0, dtype=torch.float32)

        # Forward pass through the agent
        predicted_params = agent(state_tensor)

        # Apply the transformation using the predicted parameters
        # Need to detach and convert to numpy for OpenCV
        params_np = predicted_params.detach().numpy()
        transformed_image = apply_affine_transform(image_to_correct, params_np)

        # Calculate reward from the transformation result
        reward = calculate_similarity_reward(original_image, transformed_image)

        # For reinforcement learning, we want to maximize the reward
        # So we negate it for the loss (to minimize)
        loss = -torch.tensor(reward, dtype=torch.float32)

        # Backpropagate
        optimizer.zero_grad()
        # We need to manually compute the gradients for RL
        # Use the parameters output by the network
        autograd_params = agent(state_tensor)
        # Manual loss
        rl_loss = -torch.mean(autograd_params)  # Simplified RL loss
        rl_loss.backward()  # This creates the gradients

        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

        # Debug gradients
        has_grad = False
        for name, param in agent.named_parameters():
            if param.grad is not None:
                has_grad = True
                if 'fc1.weight' in name:
                    print(f"Layer: {name}, Gradient Norm: {param.grad.norm()}")

        if not has_grad:
            print("Warning: No gradients were computed")

        # Update model parameters
        optimizer.step()

        # Save the best transformation
        if reward > best_reward:
            best_reward = reward
            best_transformed_image = transformed_image

        print(f"Epoch {epoch + 1}/{epochs}, Reward: {reward:.4f}, Loss: {loss.item():.4f}")

    print("Reinforcement Learning training finished.")
    return best_transformed_image


def main():
    image_path1 = '../../../Resources/Images/19_57_44/001.jpg'
    image_path2 = '../../../Resources/Images/19_57_44/005.jpg'
    save_path = '../../../Resources/transformed_image.jpg'
    sharper_image, to_correct_image, _, _ = load_and_compare_sharpness(image_path1, image_path2)
    transformed_image = train_affine_correction_agent(sharper_image, to_correct_image)

    cv2.imwrite(save_path, transformed_image)
    print(f"Transformed image saved as {save_path}")


if __name__ == "__main__":
    main()
