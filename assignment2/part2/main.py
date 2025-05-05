import argparse
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from model import SiameseNetwork
from dataset import FeatureMatchingDataset
from loss import ContrastiveLoss, TripletLoss
from utils import threshold_sigmoid, threshold_contrastive_loss, visualize_predictions, threshold_triplet_loss
from tqdm import tqdm

# Hyper Parameters
BATCH_SIZE = 10
NUM_EPOCHS = 20

def train(args):
    """
    Train the Siamese network
    
    Args:
        args: Command line arguments
    """
    # Define transformations resize to 256x256
    import torchvision.transforms as transforms
    default_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    train_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="train", transform=default_transform)
    print(f"Loaded {len(train_dataset)} training pairs.")
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    
    # Initialize model
    siamese_net = SiameseNetwork(contra_loss=args.contra_loss, triplet_loss=args.triplet_loss)

    if args.cuda and torch.cuda.is_available():
        siamese_net = siamese_net.cuda()
        print("Using CUDA for training")
    
    # Define loss function
    if args.triplet_loss:
        criterion = TripletLoss(margin=args.margin)
        print(f"Using Triplet Loss with margin={args.margin}")
        loss_type = 'triplet'
    elif args.contra_loss:
        criterion = ContrastiveLoss(margin=args.margin)
        print(f"Using Contrastive Loss with margin={args.margin}")
        loss_type = 'contrastive'
    else:
        criterion = torch.nn.BCELoss()
        print("Using Binary Cross Entropy Loss")
        loss_type = 'bce'

    
    # Define optimizer
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=args.lr)
    
    # Initialize lists to track metrics // CHANGED THIS TO HANDLE DIFFERENT LOSS FUNCTIONS
    train_losses = []
    
    # Train the model
    num_epochs = args.epochs
    print(f"Starting training for {num_epochs} epochs...")
    
    # ======================================================================
    # TODO: Implement the training loop
    # Your implementation should:
    # 1. Loop through all epochs
    # 2. For each epoch, iterate through the batches in train_loader
    # 3. For each batch:
    #    a. Move the data to the appropriate device (CPU/CUDA)
    #    b. Zero the parameter gradients using optimizer.zero_grad()
    #    c. Perform a forward pass through the network
    #    d. Compute the loss (different for contrastive and BCE loss)
    #    e. Perform backpropagation using loss.backward()
    #    f. Update the model parameters using optimizer.step()
    # 4. Track and print statistics (loss) for each epoch
    # 5. Periodically evaluate the model using the evaluate function
    #
    # Make sure to handle both contrastive loss and BCE loss cases appropriately
    # ======================================================================
    
    # YOUR CODE HERE

    # 1. Loop through all epochs
    for epoch in range(num_epochs):
        siamese_net.train()
        epoch_contra_loss = 0.0  # Track loss for contrastive loss
        epoch_bce_loss = 0.0  # Track loss for BCE loss
        epoch_triplet_loss = 0.0  # Track loss for triplet loss
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # 2. For each epoch, iterate through the batches in train_loader
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            labels = labels.view(-1, 1).float()

            #    a. Move the data to the appropriate device (CPU/CUDA)
            if args.cuda and torch.cuda.is_available():
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()

            #    b. Zero the parameter gradients using optimizer.zero_grad()
            optimizer.zero_grad()

            if args.triplet_loss:
                # Separate into positives and negatives
                positive_mask = (labels == 1).view(-1)
                negative_mask = (labels == 0).view(-1)

                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    anchor = img1[positive_mask]
                    positive = img2[positive_mask]
                    neg_anchor = img1[negative_mask]
                    negative = img2[negative_mask]

                    batch_size = min(anchor.size(0), negative.size(0))
                    anchor = anchor[:batch_size]
                    positive = positive[:batch_size]
                    negative = negative[:batch_size]

                    # Forward pass for triplet loss
                    out_anchor, out_pos, out_neg = siamese_net.forward_triplet(anchor, positive, negative)
                    loss = criterion(out_anchor, out_pos, out_neg)
                else:
                    continue  # Skip batch if we can't create valid triplets

            elif args.contra_loss:
                # Forward pass for contrastive loss
                output1, output2 = siamese_net(img1, img2)
                loss = criterion(output1, output2, labels)

            else:
                # Forward pass for BCE loss
                predictions = siamese_net(img1, img2)
                loss = criterion(predictions, labels)
            
            #    e. Perform backpropagation using loss.backward()
            loss.backward()

            #    f. Update the model parameters using optimizer.step()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 4. Track and print statistics (loss) for each epoch
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss ({loss_type}): {epoch_loss / len(train_loader):.4f}")

        # 5. Periodically evaluate the model using the evaluate function
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            print("------------- MODEL EVALUATION ---------------")
            test_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="test", transform=default_transform)
            print(f"Loaded {len(test_dataset)} testing pairs. (HALF OF TESTING DATASET BEING USED FOR VALIDATION)")
            half_len = len(test_dataset) // 2
            validation_subset = torch.utils.data.Subset(test_dataset, range(half_len))
            val_loader = torch.utils.data.DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)

            evaluate(args, "test", val_loader, siamese_net, visualize=False)
            print("----------------------------------------------")
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}_{loss_type}.pth"
        torch.save(siamese_net.state_dict(), checkpoint_filename)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_filename}")

        # Plot and save the loss curve at the current checkpoint
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), train_losses, label='Training Loss', color='blue') 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve ({loss_type})')
        plt.legend()
        plt.grid(True)

        # Save the plot with the epoch number
        plot_filename = f"loss_curve_epoch_{epoch+1}_{loss_type}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    
    # ======================================================================
    # END OF TODO
    # ======================================================================
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a different path based on loss type
    plot_filename = f"loss_curve_{loss_type}.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    # Save the trained model to a different path based on loss type
    model_filename = f"{args.model_file}_{loss_type}.pth"
    torch.save(siamese_net.state_dict(), model_filename)
    print(f"Saved model to {model_filename}")
    
    return siamese_net


def evaluate(args, split, data_loader, siamese_net, visualize=False):
    """
    Evaluate the Siamese network
    
    Args:
        args: Command line arguments
        split: Data split ('training' or 'testing')
        data_loader: DataLoader for the split
        siamese_net: Trained Siamese network
        visualize: Whether to visualize predictions
    """
    # Set model to evaluation mode
    siamese_net.eval()
    
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []
    sample_imgs1 = []
    sample_imgs2 = []
    
    with torch.no_grad():
        for img1_set, img2_set, labels in data_loader:
            labels = labels.view(-1, 1).float()
            
            if args.cuda and torch.cuda.is_available():
                img1_set = img1_set.cuda()
                img2_set = img2_set.cuda()
                labels = labels.cuda()
            
            # Forward pass
            if args.triplet_loss:
                # Create masks for positive and negative pairs
                positive_mask = (labels == 1).view(-1)
                negative_mask = (labels == 0).view(-1)

                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    # Ensure balanced batch sizes for anchor-positive and anchor-negative
                    anchor = img1_set[positive_mask]
                    positive = img2_set[positive_mask]
                    neg_anchor = img1_set[negative_mask]
                    negative = img2_set[negative_mask]

                    batch_size = min(anchor.size(0), negative.size(0))
                    anchor = anchor[:batch_size]
                    positive = positive[:batch_size]
                    negative = negative[:batch_size]

                    # Forward pass for triplet loss
                    out_anchor, out_pos, out_neg = siamese_net.forward_triplet(anchor, positive, negative)
                    output_labels = threshold_triplet_loss(out_anchor, out_pos, out_neg, args.margin)
                else:
                    continue  # Skip this batch if we can't create valid triplets
            else:
                # For contrastive loss and BCE loss, we follow the original behavior
                if args.contra_loss:
                    output1, output2 = siamese_net(img1_set, img2_set)
                    output_labels = threshold_contrastive_loss(output1, output2, args.margin)
                else:
                    output_labels_prob = siamese_net(img1_set, img2_set)
                    output_labels = threshold_sigmoid(output_labels_prob)
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (output_labels.view(-1) == labels.view(-1)).sum().item()

            
            # Store predictions for visualization
            if visualize and len(sample_imgs1) < 5:
                # Store a few samples for visualization
                for i in range(min(5, len(labels))):
                    if len(sample_imgs1) < 5:
                        sample_imgs1.append(img1_set[i])
                        sample_imgs2.append(img2_set[i])
                        all_labels.append(labels[i])
                        all_preds.append(output_labels[i])
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the {total} {split} images: {accuracy:.2f}%')
    
    # Visualize some predictions
    if visualize and sample_imgs1:
        visualize_predictions(
            torch.stack(sample_imgs1),
            torch.stack(sample_imgs2),
            torch.stack(all_labels),
            torch.stack(all_preds)
        )
    
    # Return model to training mode
    siamese_net.train()
    
    return accuracy


def test(args, siamese_net=None):
    """
    Test the Siamese network on the test set
    
    Args:
        args: Command line arguments
        siamese_net: Trained Siamese network (if None, load from file)
    """
    # Import transforms here to avoid circular imports
    import torchvision.transforms as transforms
    
    # Load model if not provided
    if siamese_net is None:
        siamese_net = SiameseNetwork(contra_loss=args.contra_loss, triplet_loss=args.triplet_loss)
        siamese_net.load_state_dict(torch.load(args.model_file))
        print(f"Loaded model from {args.model_file}")
    
    if args.cuda and torch.cuda.is_available():
        siamese_net = siamese_net.cuda()
    
    # Define transformations
    default_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load test dataset
    test_dataset = FeatureMatchingDataset(args.data_dir, args.train_file, split="test", transform=default_transform)
    print(f"Loaded {len(test_dataset)} testing pairs.")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Evaluate on test set
    test_acc = evaluate(args, "testing", test_loader, siamese_net, visualize=True)
    
    return test_acc


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Siamese Network for Feature Matching')
    parser.add_argument('--action', type=str, choices=['train', 'test', 'train_test'], 
                        default='train_test', help='Action to perform')
    parser.add_argument('--data_dir', type=str, default='./', 
                        help='Directory containing training images')
    parser.add_argument('--train_file', type=str, default='./groundtruth.json', 
                        help='File containing training pairs') 
    parser.add_argument('--model_file', type=str, default='siamese_model.pth', 
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--margin', type=float, default=1.0, 
                        help='Margin for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='Use CUDA if available')
    parser.add_argument('--contra_loss', action='store_true', default=False, 
                        help='Use contrastive loss instead of BCE')
    parser.add_argument('--eval_freq', type=int, default=1, 
                        help='Frequency of evaluation during training')
    parser.add_argument('--triplet_loss', action='store_true', default=False, 
                    help='Use triplet loss instead of BCE or contrastive loss')
    
    args = parser.parse_args()
    
    print(f"Running with arguments: {args}")
    
    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    elif args.action == 'train_test':
        siamese_net = train(args)
        test(args, siamese_net)


if __name__ == '__main__':
    main()