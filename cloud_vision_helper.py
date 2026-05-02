from google.cloud import vision


def cloud_classify_image(image_path: str):
    """
    Use Google Cloud Vision label detection on an image file.
    Returns a list of (label, score_percent).
    """
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    if response.error.message:
        raise RuntimeError(response.error.message)

    labels = []
    for label in response.label_annotations[:5]:
        labels.append((label.description, round(label.score * 100, 1)))

    return labels