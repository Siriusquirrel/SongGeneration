import torch

def split_checkpoint(checkpoint_path: str):
    print(f"Loading checkpoint via mmap: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)

    ht_state_dict = {}
    st_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("transformer.") or key.startswith("emb.") or key.startswith("condition_provider."):
            ht_state_dict[key] = tensor
        # Der Sub-Transformer und seine exklusiven Module auf oberster Ebene
        elif (key.startswith("transformer2.") or 
              key.startswith("layer2_emb.") or 
              key.startswith("linears.") or 
              key.startswith("mlp.")):
            st_state_dict[key] = tensor
        else:
            ht_state_dict[key] = tensor

    print("Writing HT checkpoint ...")
    torch.save(ht_state_dict, "model_ht_v2_large_fp16_new_data_structure.pt")

    print("Writing ST checkpoint ...")
    torch.save(st_state_dict, "model_st_v2_large_fp16_new_data_structure.pt")

    print("Finished!")

if __name__ == "__main__":
    split_checkpoint("model_v2_large_fp16_new_data_structure.pt")
