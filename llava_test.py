from model_llava import classify_with_model

def main():
    image_path = "/home/tim/ICL-VL/data/tumor/all/N1.jpg"  # Replace this with your image path

    # Try LLaVA
    print("🔹 LLaVA result:")
    result_llava = classify_with_model(
        model_type="llava",
        image_path=image_path,
        prompt="Please describe this medical image."
    )
    print(result_llava)


    # Try LLaVA-Med
    print("\n🔹 LLaVA-Med result:")
    result_llava_med = classify_with_model(
        model_type="llava-med",
        image_path=image_path,
        prompt="Please provide a diagnostic impression for this radiology image."
    )
    print(result_llava_med)

if __name__ == "__main__":
    main()