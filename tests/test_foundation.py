from src.data.text_template import build_text

def test_build_text():
    product = {
        "gender": "Men",
        "usage": "Casual",
        "articleType": "Shirts",
        "baseColour": "Navy Blue",
        "season": "Fall",
        "productDisplayName": "Checked Cotton Shirt",
    }

    text = build_text(product)

    assert text == (
        "Checked Cotton Shirt. "
        "Men Casual Shirts in Navy Blue for Fall"
    )