import torch
from torch.utils.data import DataLoader

import config
from dataset import load_and_split_data, NiftiXRVDataset
from model import XRVHeightRegressor
from utils import run_epoch, visualize_sample,visualize_dataset_samples

def main():



    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_df, val_df, test_df = load_and_split_data()

    train_dataset = NiftiXRVDataset(train_df, training=True)
    val_dataset = NiftiXRVDataset(val_df, training=False)
    test_dataset = NiftiXRVDataset(test_df, training=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)


    print("Visualizing Training Data (With Augmentations):")
    visualize_dataset_samples(train_dataset, num_samples=5,title_prefix="Train Set")


    # 2. Setup Model
    model = XRVHeightRegressor(spacing_dim=2).to(config.DEVICE)

    # 3. Stage 1: Train Head Only
    print("\n--- Stage 1: Training Head Only ---")
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]
    optimizer = torch.optim.Adam(head_params, lr=1e-3)

    best_val_mae, patience, bad = float('inf'), 8, 0
    save_path_head = config.EXPERIMENTS_DIR / "best_xrv_head.pt"

    for epoch in range(1, 40):
        tr_mse, tr_mae = run_epoch(model, train_loader, optimizer)
        va_mse, va_mae = run_epoch(model, val_loader, optimizer=None)
        print(f"[E{epoch:02d}] Train MAE {tr_mae:.2f} || Val MAE {va_mae:.2f}")

        if va_mae < best_val_mae - 1e-6:
            best_val_mae, bad = va_mae, 0
            torch.save(model.state_dict(), save_path_head)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # 4. Stage 2: Fine Tuning
    print("\n--- Stage 2: Fine Tuning Backbone ---")
    model.load_state_dict(torch.load(save_path_head, map_location=config.DEVICE))

    # Unfreeze specific layers
    for name, p in model.backbone.named_parameters():
        p.requires_grad = ('denseblock4' in name) or ('norm5' in name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_val_mae_ft, bad = best_val_mae, 0
    save_path_ft = config.EXPERIMENTS_DIR / "best_xrv_finetuned.pt"

    for epoch in range(1, 10):
        tr_mse, tr_mae = run_epoch(model, train_loader, optimizer)
        va_mse, va_mae = run_epoch(model, val_loader, optimizer=None)
        print(f"[FT{epoch:02d}] Train MAE {tr_mae:.2f} || Val MAE {va_mae:.2f}")
        if va_mae < best_val_mae_ft - 1e-6:
            best_val_mae_ft, bad = va_mae, 0
            torch.save(model.state_dict(), save_path_ft)
        else:
            bad += 1
            if bad >= patience:
                break

    # 5. Final Evaluation & Visualization
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(save_path_ft, map_location=config.DEVICE))
    model.eval()

    test_mse, test_mae = run_epoch(model, test_loader, optimizer=None)
    print(f"Final Test MAE: {test_mae:.2f} cm")

    # Run Visualization
    try:
        visualize_sample(model, test_dataset, test_df)
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    main()