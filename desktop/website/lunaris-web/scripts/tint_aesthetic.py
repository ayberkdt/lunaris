import os
from PIL import Image, ImageEnhance, ImageOps

def create_maps():
    print("Loading 8k_moon.jpg...")
    img = Image.open('../public/textures/8k_moon.jpg').convert('RGB')
    img = img.resize((4096, 2048), Image.Resampling.LANCZOS)
    width, height = img.size
    
    # Premium Aesthetic Form (Sleek, sharp, high-contrast titanium look with subtle blue/purple tint)
    print("Enhancing for Premium Aesthetic...")
    
    # 1. Contrast & Sharpness
    img_aesthetic = ImageEnhance.Contrast(img).enhance(1.4)
    img_aesthetic = ImageEnhance.Sharpness(img_aesthetic).enhance(1.5)
    img_aesthetic = ImageEnhance.Brightness(img_aesthetic).enhance(0.9)
    
    # 2. Sleek Tint (Titanium / Slate Blue)
    # We want shadows to be deep rich black, midtones to be cool metallic gray, highlights pure white.
    pixels = img_aesthetic.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            intensity = (r + g + b) / 3.0
            
            # Subtle deep slate/blue tinting
            if intensity < 100:
                # Dark areas: Deep midnight blue
                tr = int(r * 0.8)
                tg = int(g * 0.9)
                tb = int(b * 1.3)
            elif intensity < 180:
                # Midtones: Cool titanium gray
                tr = int(r * 0.95)
                tg = int(g * 0.95)
                tb = int(b * 1.1)
            else:
                # Highlights: Crisp white/cyan
                tr = min(255, int(r * 1.05))
                tg = min(255, int(g * 1.05))
                tb = min(255, int(b * 1.1))
                
            pixels[x, y] = (tr, tg, tb)

    print("Saving aesthetic_moon_real.jpg...")
    img_aesthetic.save('../public/textures/aesthetic_moon_real.jpg', quality=95)
    
    print("Done!")

if __name__ == '__main__':
    create_maps()
