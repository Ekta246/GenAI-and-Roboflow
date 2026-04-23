"""
Fire Detection Image Generator using Grok AI (xAI API)
Generates synthetic fire images from security surveillance camera perspectives.

Focus:
    - FIRE images: Bright visible flames (primary subject)
    - FIRE with smoke: Optional compact localized smoke (30% of fire images)
    - SMOKE-ONLY images: Compact smoke without visible fire (25% of all images)
    - All smoke: LOCALIZED and contained - NO spread-out room-filling smoke
    - Easy to annotate: Tight vertical columns/plumes for bounding boxes
    - Security surveillance camera angles (ceiling/wall-mounted CCTV)
    - Indoor AND outdoor locations (60% indoor, 40% outdoor)
    - Updated location variety for diverse environments
    
Annotation Workflow:
    1. Generate images with this script
    2. Run inference with your YOLO model (yolo-inf.py)
    3. Get annotations in Pascal VOC format automatically
    
Requirements:
    pip install openai  # xAI uses OpenAI-compatible API
    
Setup:
    1. Get your API key from: https://console.x.ai/
    2. Set environment variable: export XAI_API_KEY="your-key-here"
    3. Or set it directly in CONFIG below
"""

import os
import requests
import json
from pathlib import Path
import random
import time
from datetime import datetime
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ===================== CONFIGURATION =====================
CONFIG = {
    # API ConfigurationYOUR_XAI_API_KEY_HERE
    "api_key": os.getenv("XAI_API_KEY", ""),  # Set via env/.env
    "api_base": "https://api.x.ai/v1",  # xAI API endpoint
    
    # Output settings
    "output_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/sf-images",
    "num_images": 300,  # Number of images to generate
    
    # Generation focus
    "fire_only": True,  # Primary focus on fire
    "compact_smoke_probability": 0.4,  # 30% chance of compact localized smoke WITH fire
    "smoke_only_probability": 0.45,  # 65% smoke-only images, 35% fire images
    "indoor_outdoor_ratio": 0.65,  # 20% indoor, 80% outdoor (favor outdoor bright scenarios)
    "include_outdoor": True,  # Include outdoor fire scenarios
    
    # Generation settings
    "image_size": "1280x720",  # Note: xAI doesn't support size parameter yet (uses default)
    "delay_between_requests": 2,  # Seconds between API calls
    
    # Model settings
    "model": "grok-imagine-image",
      # Image generation model
    "text_model": "grok-beta",  # Text model for prompt enhancement
    "temperature": 0.9,  # Higher = more creative (0.0 to 1.0)

    # Composite mode: paste fire/smoke crops onto your backgrounds (disabled — using Grok direct generation)
    "composite_mode": False,  # Enable composite image generation
    "composite_count": 100,  # How many composite images to generate
    "backgrounds_dir": "/Users/ekta/Downloads/smoke-fire-dataset/backgrounds_agri",
    "fire_sources_dir": "/Users/ekta/Downloads/smoke-fire-dataset/fire-images",
    "fire_annotations_dir": "/Users/ekta/Downloads/smoke-fire-dataset/fire-annotations",
    "objects_per_image": (1, 3),  # Random 1 to 3 fire/smoke objects per composite
    "composite_scale_range": (0.08, 0.25),  # Small flames to max ~1/4 of image area
    "grok_crop_count": 15,  # Number of isolated fire/smoke crops to generate with Grok (60% fire, 40% smoke)
    "output_size": (1280, 720),
}

# ===================== PROMPT TEMPLATES =====================

INDOOR_LOCATIONS = [
    "abandoned textile mill with rusted looms and fabric scraps",
    "server room with blinking rack-mounted equipment and cable trays",
    "underground metro maintenance tunnel with exposed pipes",
    "veterinary clinic storage with oxygen cylinders",
    "bowling alley mechanical pit behind the lanes",
    "indoor go-kart track pit area with fuel cans",
    "church attic with old wooden beams and insulation",
    "library rare book archive with floor-to-ceiling shelving",
    "indoor swimming pool pump and chemical room",
    "brewery fermentation room with steel vats",
    "commercial greenhouse with grow lights and plastic sheeting",
    "self-storage facility corridor with metal roll-up doors",
    "elevator machine room on building rooftop",
    "backstage area of a theater with curtains and props",
    "prison laundry facility with industrial washers",
    "ice rink compressor room with ammonia lines",
    "apartment building trash chute room",
    "hotel boiler room in the basement",
    "shooting range ventilation corridor",
    "electronics recycling warehouse with circuit board bins",
    "indoor fish market cold storage area",
    "nightclub DJ booth and backstage area",
    "dental clinic sterilization room",
]

OUTDOOR_LOCATIONS = [
    "solar panel farm with inverter stations",
    "highway overpass underside near concrete pillars",
    "outdoor flea market with canvas tent stalls",
    "marina dock with wooden pier and fuel pumps",
    "abandoned drive-in movie theater lot",
    "cell tower base station with equipment cabinets",
    "grain elevator complex with conveyor belts",
    "outdoor recycling center with baled material",
    "water treatment plant outdoor chemical tanks",
    "ski resort base lodge exterior with stored equipment",
    "fairground midway with carnival ride structures",
    "agricultural feed lot with hay bale stacks",
    "oil pipeline valve station in open field",
    
    "outdoor tire storage yard",
    "abandoned gas station with rusted pumps",
    "airport tarmac near ground service equipment",
    "rooftop of a multi-story parking garage",
   
    "outdoor transformer substation with high-voltage gear",
    "construction crane staging area",
    "lumber yard with stacked timber",
    
    "campground fire pit area surrounded by trees",
]

FIRE_SHAPES = [
    "tiny flickering flame barely visible, like a candle just lit",
    "small sharp tongues of fire licking upward from one point",
    "medium ragged fire with irregular jagged edges spreading sideways",
    "tall narrow column of flame shooting straight up like a torch",
    "wide low-spreading ground fire creeping across a surface",
    "explosive fireball shape with radiating orange edges",
    "swirling vortex of flame twisting upward like a tornado",
    "multiple small separate fire patches scattered across an area",
    "lazy rolling flames with slow undulating motion",
    "bright white-hot core surrounded by orange and red outer flames",
    "sputtering sparking fire with glowing embers flying outward",
    "thin sheet of blue-orange flame running along a surface edge",
    "mushroom-shaped fire plume billowing outward at the top",
    "crackling fire with visible sparks popping in the air",
    "smoldering glow with intermittent small flame bursts",
    "roaring blaze with thick turbulent flames reaching high",
]

SMOKE_SHAPES = [
    "thin pencil-like column of white smoke drifting straight up",
    "thick black mushroom-shaped smoke cloud rising and spreading at top",
    "wispy translucent gray smoke curling and twisting in the air",
    "dense oily black smoke hugging the ceiling in a flat layer",
    "pulsing bursts of smoke coming in rhythmic puffs",
    "spiraling corkscrew-shaped smoke rising in a tight helix",
    "heavy low-hanging white fog-like smoke clinging to the ground",
    "dark charcoal smoke with visible particulate texture",
    "fast-rising jet of gray smoke like a chimney exhaust",
    "thin horizontal smoke trail drifting sideways in a breeze",
    "billowing cumulus-shaped gray smoke expanding as it rises",
    "sharp-edged dark smoke plume with clearly defined boundaries",
    "layered smoke with darker core and lighter transparent edges",
    "slow lazy smoke barely moving, hanging in still air",
    "acrid yellowish-brown chemical smoke with unusual color",
    "intermittent smoke appearing and disappearing in gusts",
]

FIRE_SOURCES = [
    "lithium battery pack swelling and venting flames",
    "welding sparks igniting oil-soaked rags",
    "overheated industrial motor bearing",
    "arc flash from corroded electrical switchgear",
    "spontaneous combustion of oily shop towels in a bin",
    "gas leak igniting near a valve fitting",
    "friction fire from seized conveyor belt roller",
    "lightning strike damage on rooftop equipment",
    "transformer oil leak catching fire",
    "static discharge igniting solvent vapor",
    "propane tank valve failure near a grill",
    "dry grass igniting from heat near metal equipment",
    "shorted industrial charging cable melting and burning",
    "dryer lint buildup catching fire in exhaust duct",
    "chemical reaction between improperly stored materials",
    "cigarette butt smoldering in mulch bed",
    "bird nest in electrical junction box shorting wires",
    "hydraulic line burst spraying hot oil onto hot surface",
    "sun focused through glass onto combustible material",
    "overloaded extension cord melting under carpet",
]

LIGHTING_CONDITIONS = [
    "harsh midday sun casting sharp shadows",
    "golden hour warm light from low sun angle",
    "overcast flat daylight with no shadows",
    "bright fluorescent tubes with slight green tint",
    "sodium vapor orange glow from parking lot lights",
    "LED panel white light evenly distributed",
    "mixed daylight and artificial light creating dual shadows",
    "dusk blue ambient light with security lights turning on",
    "bright halogen work lights flooding the area",
    "dawn gray light with first hints of sunrise",
    "moonlight with motion-activated LED floodlights",
    "emergency red strobe lighting mixed with normal lights",
    "indirect window light bouncing off white walls",
    "industrial high-bay metal halide lighting from above",
    "cloudy bright daylight diffused and even",
]

SURVEILLANCE_PERSPECTIVES = [
    "ceiling-mounted dome camera looking straight down at 90 degrees, bird's eye view",
    "corner-mounted PTZ camera tilted 45 degrees downward, capturing full room diagonal",
    "wall-mounted bullet camera at 3 meter height angled 30 degrees down",
    "outdoor pole-mounted camera at 5 meters looking down at parking area",
    "fisheye ceiling camera with 180-degree hemispherical distortion",
    "recessed ceiling camera with narrow 60-degree field of view, zoomed in",
    "exterior wall-mounted camera under eave, looking across open area",
    "stairwell corner camera angled down steep stairs",
    "elevator interior ceiling camera looking down at floor",
    "loading dock overhead camera capturing bay doors and empty dock area",
    "hallway end-mounted camera with long corridor perspective vanishing point",
    "entrance vestibule camera at door frame height capturing both sides",
    "high-mounted camera on warehouse rafter looking down 20 meters",
    "parking garage low-ceiling camera with wide distorted angle",
    "perimeter fence camera on post looking along fence line",
    "rooftop camera angled down building facade at street below",
    "indoor corner camera capturing two perpendicular walls and floor",
    "tunnel camera with long depth perspective and harsh lighting",
]

CAMERA_DETAILS = [
    "4K IP security camera with slight motion blur",
    "grainy 720p analog CCTV with scan lines",
    "crisp 1080p digital surveillance camera",
    "infrared-capable camera in daytime color mode",
    "wide dynamic range camera balancing bright and dark areas",
    "varifocal lens security camera with slight barrel distortion",
    "weatherproof outdoor camera with water droplets on lens edge",
    "dome camera with slight tinted dome reflection visible",
    "PTZ camera frozen mid-pan with slight motion streak",
    "timestamp-overlaid security footage with date and camera ID",
]

# ===================== PROMPT GENERATION =====================

def generate_fire_prompt() -> str:
    """Generate a creative fire scene prompt from security surveillance perspective."""
    is_indoor = random.random() < (1 - CONFIG["indoor_outdoor_ratio"])
    location = random.choice(INDOOR_LOCATIONS if is_indoor else OUTDOOR_LOCATIONS)
    fire_shape = random.choice(FIRE_SHAPES)
    fire_source = random.choice(FIRE_SOURCES)
    lighting = random.choice(LIGHTING_CONDITIONS)
    perspective = random.choice(SURVEILLANCE_PERSPECTIVES)
    camera = random.choice(CAMERA_DETAILS)

    smoke_addition = ""
    if random.random() < CONFIG["compact_smoke_probability"]:
        smoke_shape = random.choice(SMOKE_SHAPES)
        smoke_addition = f"Above the fire: {smoke_shape}, staying compact directly above the flame source."

    num_fires = random.choice(["single fire source", "two separate small fires", "three distinct fire spots at different locations"])

    prompt = f"""Photorealistic {camera} captured at 1280x720 resolution.
Scene: {location}, viewed from {perspective}.
Lighting: {lighting}. ABSOLUTELY NO people, NO vehicles, NO cars, NO trucks in the scene.
FIRE: {num_fires}. Flame appearance: {fire_shape}.
Cause: {fire_source}. Fire occupies less than one quarter of the frame.
{smoke_addition}
Style: Authentic security surveillance footage with slight wide-angle lens distortion.
Timestamp overlay in corner showing date and camera ID.
The fire is the focal anomaly in an otherwise empty monitored environment.
CRITICAL: The scene must be completely empty of humans and vehicles. Only the environment and fire/smoke.
Image must look like a real frame grabbed from a CCTV recording system.
Widescreen 16:9 aspect ratio. Realistic depth and perspective."""
    return prompt.strip()


def generate_smoke_prompt() -> str:
    """Generate a creative smoke-only scene prompt from security surveillance perspective."""
    is_indoor = random.random() < (1 - CONFIG["indoor_outdoor_ratio"])
    location = random.choice(INDOOR_LOCATIONS if is_indoor else OUTDOOR_LOCATIONS)
    smoke_shape = random.choice(SMOKE_SHAPES)
    lighting = random.choice(LIGHTING_CONDITIONS)
    perspective = random.choice(SURVEILLANCE_PERSPECTIVES)
    camera = random.choice(CAMERA_DETAILS)

    smoke_origin = random.choice([
        "seeping from behind a closed panel",
        "rising from an unseen source below frame",
        "emerging from a crack in equipment housing",
        "leaking from a ceiling vent or duct",
        "escaping from a gap in wall-mounted infrastructure",
        "curling upward from an electrical conduit",
        "venting from overheated machinery internals",
        "wafting from a concealed smoldering source",
        "trickling out from a storage cabinet seam",
    ])

    prompt = f"""Photorealistic {camera} captured at 1280x720 resolution.
Scene: {location}, viewed from {perspective}.
Lighting: {lighting}. ABSOLUTELY NO people, NO vehicles, NO cars, NO trucks in the scene.
SMOKE ONLY — absolutely no visible flames or fire.
Smoke appearance: {smoke_shape}.
Smoke origin: {smoke_origin}.
The smoke is compact and localized — it does NOT fill the room or spread across the scene.
Smoke occupies a small well-defined area, easy to draw a bounding box around.
The rest of the environment is clearly visible and unobstructed.
Style: Authentic security surveillance footage with slight wide-angle lens distortion.
The smoke is the only anomaly in an otherwise empty monitored space.
CRITICAL: The scene must be completely empty of humans and vehicles. Only the environment and smoke.
Image must look like a real frame grabbed from a CCTV recording system.
Widescreen 16:9 aspect ratio. Realistic depth and perspective."""
    return prompt.strip()


def generate_specific_scenario_prompts() -> list:
    """Hand-crafted unique high-priority scenarios for maximum diversity."""
    return [
        # --- FIRE: unusual / creative scenarios ---
        "4K security camera still frame 1280x720: tiny flickering flame just starting on a lithium battery pack on a metal shelf inside a self-storage unit, ceiling dome camera looking straight down, bright LED overhead lights, the fire is small but bright orange against gray metal surroundings, no people, no vehicles, timestamp overlay, authentic CCTV look",

        "Grainy 720p CCTV frame: medium fire with jagged irregular flames spreading along a wooden pallet stack at an outdoor lumber yard, pole-mounted camera 6 meters high looking down, harsh midday sun, fire occupies one corner of frame, smoke is minimal and tight above flames, no people, no vehicles, surveillance footage aesthetic",

        "HD surveillance screenshot 1280x720: bright white-hot electrical arc fire erupting from a corroded outdoor transformer substation, wall-mounted bullet camera angle, golden hour warm light, sparks radiating outward, compact fire contained to the transformer unit, no people, no vehicles, slight lens barrel distortion",

        "Digital security camera 1280x720: swirling flame vortex from a grease trap fire behind a restaurant exhaust hood, ceiling corner camera 45 degree angle, fluorescent kitchen lighting, fire spiraling upward in a tight column, stainless steel surroundings reflecting orange glow, no people, no vehicles, CCTV timestamp",

        "IP camera frame 1280x720: two separate small fires burning simultaneously — one on a cardboard pile and one from an electrical outlet — inside an electronics recycling warehouse, overhead fisheye camera, industrial high-bay lighting, fires are small and clearly separated, no people, no vehicles",

        "Security camera capture 1280x720: crackling fire with visible sparks from a bird nest shorting wires inside an exterior electrical junction box, wall-mounted outdoor camera, overcast flat daylight, compact fire against beige building wall, sparks falling downward, no people, no vehicles, weatherproof camera look",

        "CCTV frame grab 1280x720: large roaring blaze with turbulent flames from a propane tank valve failure at an outdoor campground fire pit, perimeter fence camera on post, dusk blue light with security floods turning on, dramatic orange fire against darkening sky, no people, no vehicles",

        # --- SMOKE: unusual / creative scenarios ---
        "4K IP camera still 1280x720: thin pencil-like column of white smoke rising from behind a closed server rack panel in a data center, ceiling dome camera looking down, blue-tinted LED ambient light with blinking status LEDs, smoke thin and perfectly vertical, rest of room clear, no people, no vehicles, timestamp",

        "HD surveillance 1280x720: spiraling corkscrew-shaped gray smoke rising from overheated conveyor belt machinery in an abandoned textile mill, corner PTZ camera, mixed window daylight and fluorescent, smoke twisting upward in a helix shape, localized to one machine, no fire, no people, no vehicles",

        "Digital security camera 1280x720: dark charcoal smoke with visible particulate billowing from a chemical storage cabinet inside an indoor swimming pool pump room, wall-mounted camera at 3 meters, harsh overhead fluorescent, smoke dense and dark against white walls, no flames, no people, no vehicles",

        "IP camera 1280x720: fast-rising jet of gray smoke like chimney exhaust shooting upward from a drainage grate at a railroad switching yard, elevated outdoor camera on pole, bright midday sun, smoke column sharp and well-defined against blue sky, no fire visible, no people, no vehicles",

        "Surveillance screenshot 1280x720: acrid yellowish-brown chemical smoke seeping from a cracked pipe fitting at a water treatment plant outdoor tank area, perimeter camera, cloudy bright daylight, unusual smoke color clearly visible against industrial gray background, compact plume, no fire, no people, no vehicles",

        "CCTV frame 1280x720: layered smoke with darker core and lighter transparent edges drifting from an elevator machine room doorway, hallway end-mounted camera with vanishing point perspective, emergency lighting mixing with normal corridor lights, smoke hugging ceiling in defined band, no fire, no people, no vehicles",

        "Security camera 1280x720: intermittent pulsing bursts of white smoke appearing and disappearing from a brewery fermentation vat pressure release valve, ceiling-mounted camera looking down, industrial lighting, each smoke burst compact and dissipating quickly, no fire, no people, no vehicles, authentic footage look",
    ]


# ===================== xAI API FUNCTIONS =====================

def setup_xai_client():
    """Initialize xAI API client."""
    api_key = CONFIG["api_key"]
    
    if not api_key:
        raise ValueError(
            "\n❌ No API key found!\n"
            "Please set your xAI API key:\n"
            "  1. Get key from: https://console.x.ai/\n"
            "  2. Set environment variable: export XAI_API_KEY='your-key'\n"
            "  3. Or set it in .env as XAI_API_KEY=your-key\n"
        )
    
    # xAI uses OpenAI-compatible API
    client = OpenAI(
        api_key=api_key,
        base_url=CONFIG["api_base"]
    )
    
    return client


def generate_image_with_grok(prompt: str, output_path: str) -> bool:
    """
    Generate image using xAI Grok Imagine API.
    Model: grok-imagine-image
    """
    try:
        client = setup_xai_client()
        
        print(f"  🎨 Generating with {CONFIG['model']}...")
        print(f"  📝 Prompt: {prompt[:80]}...")
        
        try:
            # Generate image with grok-imagine-image model
            response = client.images.generate(
                model=CONFIG["model"],
                prompt=prompt,
                n=1,
            )
            
            # Download and save image
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            print(f"  ✅ Image generated successfully!")
            return True
            
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"  ❌ API Error: {error_msg}")
            
            # Provide helpful debugging info
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"  💡 Model '{CONFIG['model']}' not found. Try:")
                print(f"     - grok-imagine-image")
                print(f"     - grok-vision-beta")
                print(f"     - Check https://console.x.ai/docs for latest models")
            elif "403" in error_msg or "permission" in error_msg.lower():
                print(f"  💡 Check billing: https://console.x.ai/")
            
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def enhance_prompt_with_grok(basic_prompt: str) -> str:
    """
    Use Grok's text capabilities to enhance and refine the prompt.
    This can be used with other image generation services.
    """
    try:
        client = setup_xai_client()
        
        enhancement_request = f"""Enhance this image generation prompt to be more detailed and photorealistic:

"{basic_prompt}"

Make it more specific with:
- Precise camera details (angle, distance, focal length)
- Exact lighting conditions and color temperature
- Specific material textures and surfaces
- Realistic fire/smoke physics
- Professional safety documentation style

Return only the enhanced prompt, no explanations."""

        response = client.chat.completions.create(
            model=CONFIG["text_model"],  # Use text model for prompt enhancement
            messages=[
                {"role": "system", "content": "You are an expert at creating detailed prompts for photorealistic image generation, specializing in fire safety and emergency scenarios."},
                {"role": "user", "content": enhancement_request}
            ],
            temperature=CONFIG["temperature"]
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        return enhanced_prompt
        
    except Exception as e:
        print(f"  ⚠️  Could not enhance prompt: {e}")
        return basic_prompt


# ===================== GROK-GENERATED CROPS + COMPOSITE =====================

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Prompts for generating isolated fire/smoke on dark backgrounds
ISOLATED_FIRE_PROMPTS = [
    "Realistic bright orange and yellow fire flames on a pure black background. Isolated fire, no surroundings. Flames rising upward with natural flickering shapes.",
    "Small campfire-sized flames on a solid black background. Bright orange fire, isolated, no environment. Clean edges visible against the dark.",
    "Large intense fire with red and orange flames on a black background. Isolated fire element, dramatic and bright, no scene context.",
    "Realistic flickering fire flames on a dark black background. Medium-sized fire, natural fire shape, warm orange glow. Isolated object only.",
    "Bright yellow and orange fire burning on pure black background. Small to medium flames, natural look, isolated fire element for compositing.",
    "Intense raging fire with sparks on solid black background. Large flames reaching upward, bright orange and red. Isolated fire only.",
    "Gentle small fire flame on pure black background. Single flame source, warm orange, natural flickering. Clean isolated element.",
    "Wildfire-style flames on black background. Wide spreading fire, bright orange-red, natural turbulent flame shape. Isolated.",
    "Industrial fire with thick bright flames on solid black background. Orange and yellow fire, medium size, isolated element.",
    "Torch-like fire flame on pure black background. Tall narrow flame, bright orange-yellow, isolated fire element.",
]

ISOLATED_SMOKE_PROMPTS = [
    "Realistic gray smoke column rising on a pure black background. Compact vertical smoke plume, dense at base, fading at top. Isolated smoke only.",
    "Thick dark gray smoke on solid black background. Compact localized smoke cloud, not spread out. Dense and visible. Isolated element.",
    "Light wispy smoke trail on pure black background. Thin vertical smoke column, slightly curling. Natural smoke behavior. Isolated.",
    "Dense white-gray smoke plume on black background. Compact column rising upward, well-defined edges. Isolated smoke element.",
    "Dark smoke billowing upward on solid black background. Medium density, compact shape, not spread across image. Isolated element.",
    "Industrial-style thick gray smoke on pure black background. Dense compact plume, rising vertically. Well-defined shape. Isolated.",
    "Thin smoke wisps on black background. Light gray, delicate curling smoke. Small and contained. Isolated smoke element.",
    "Heavy dark smoke column on solid black background. Dense and opaque at center, natural edges. Compact and localized. Isolated.",
]


def generate_grok_fire_smoke_crops(num_crops: int = 15) -> list:
    """Generate isolated fire/smoke images using Grok AI, then extract the bright regions as crops."""
    print(f"\n  🎨 Generating {num_crops} fire/smoke crops with Grok AI...")

    crops_dir = Path(CONFIG["output_dir"]) / "grok_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    client = setup_xai_client()

    fire_count = int(num_crops * 0.6)
    smoke_count = num_crops - fire_count

    all_prompts = []
    for i in range(fire_count):
        all_prompts.append(("fire", random.choice(ISOLATED_FIRE_PROMPTS)))
    for i in range(smoke_count):
        all_prompts.append(("smoke", random.choice(ISOLATED_SMOKE_PROMPTS)))

    random.shuffle(all_prompts)

    for idx, (obj_class, prompt) in enumerate(all_prompts):
        print(f"    [{idx+1}/{num_crops}] Generating {obj_class} crop...")
        try:
            response = client.images.generate(
                model=CONFIG["model"],
                prompt=prompt,
                n=1,
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content

            # Save the raw generated image
            raw_path = str(crops_dir / f"grok_{obj_class}_{idx:03d}.png")
            with open(raw_path, 'wb') as f:
                f.write(image_data)

            # Load and extract bright region (fire/smoke on dark bg)
            img = cv2.imread(raw_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold to find the fire/smoke (bright pixels against dark background)
            if obj_class == "fire":
                _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            else:
                _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

            # Find bounding box of the bright region
            coords = cv2.findNonZero(mask)
            if coords is None:
                print(f"      Warning: No {obj_class} region found, skipping")
                continue

            x, y, w, h = cv2.boundingRect(coords)
            if w < 20 or h < 20:
                continue

            crop = img[y:y+h, x:x+w].copy()
            crops.append({"image": crop, "class": obj_class, "source": f"grok_{obj_class}_{idx:03d}.png"})
            print(f"      Extracted {obj_class} crop: {w}x{h}px")

        except Exception as e:
            print(f"      Error generating crop: {e}")
            continue

        time.sleep(CONFIG["delay_between_requests"])

    print(f"  Generated {len(crops)} usable fire/smoke crops")
    return crops


def load_backgrounds_for_composite() -> list:
    """Load all background images."""
    bg_dir = Path(CONFIG["backgrounds_dir"])
    backgrounds = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for bg_path in bg_dir.glob(ext):
            bg = cv2.imread(str(bg_path))
            if bg is not None:
                backgrounds.append(bg)
    return backgrounds


def paste_crop_on_background(bg: np.ndarray, crop: np.ndarray, position: tuple, scale: float) -> tuple:
    """Paste a crop onto a background at given position and scale. Returns (composite, bbox)."""
    new_h = int(crop.shape[0] * scale)
    new_w = int(crop.shape[1] * scale)
    if new_h < 10 or new_w < 10:
        return bg, None

    resized = cv2.resize(crop, (new_w, new_h))
    y, x = position
    bg_h, bg_w = bg.shape[:2]

    y1, x1 = max(0, y), max(0, x)
    y2, x2 = min(bg_h, y + new_h), min(bg_w, x + new_w)
    cy1, cx1 = y1 - y, x1 - x
    cy2, cx2 = cy1 + (y2 - y1), cx1 + (x2 - x1)

    if y2 <= y1 or x2 <= x1:
        return bg, None

    # Alpha blend: use brightness of crop as alpha for natural blending on dark-bg crops
    crop_region = resized[cy1:cy2, cx1:cx2].astype(np.float32)
    bg_region = bg[y1:y2, x1:x2].astype(np.float32)
    gray = cv2.cvtColor(crop_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    alpha = (gray.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    blended = (crop_region * alpha_3ch + bg_region * (1 - alpha_3ch)).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended

    return bg, (x1, y1, x2, y2)


def save_composite_voc_xml(image_path: str, img_shape: tuple, objects: list, xml_path: str):
    """Save Pascal VOC XML annotation for a composite image."""
    h, w = img_shape[:2]
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"

    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["bbox"][0])
        ET.SubElement(bndbox, "ymin").text = str(obj["bbox"][1])
        ET.SubElement(bndbox, "xmax").text = str(obj["bbox"][2])
        ET.SubElement(bndbox, "ymax").text = str(obj["bbox"][3])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)


def generate_composite_images():
    """Generate composite images: Grok AI fire/smoke crops + your backgrounds."""
    print(f"\n🖼️  Grok AI Crop + Composite Mode")
    print(f"{'='*60}")
    print(f"  Step 1: Generate isolated fire/smoke with Grok AI")
    print(f"  Step 2: Extract fire/smoke regions from dark backgrounds")
    print(f"  Step 3: Alpha-blend onto your backgrounds")
    print(f"{'='*60}")

    # Step 1: Generate fire/smoke crops with Grok
    grok_crops = generate_grok_fire_smoke_crops(num_crops=CONFIG.get("grok_crop_count", 15))

    if not grok_crops:
        print("  ❌ No Grok crops generated. Check API key and connection.")
        return 0

    # Step 2: Load your backgrounds
    backgrounds = load_backgrounds_for_composite()
    if not backgrounds:
        print("  ❌ No backgrounds found. Check backgrounds_dir.")
        return 0

    print(f"\n  🖼️  Compositing onto {len(backgrounds)} backgrounds...")
    print(f"  Fire/smoke crops available: {len(grok_crops)}")
    print(f"  Target images: {CONFIG['composite_count']}")
    print(f"  Objects per image: {CONFIG['objects_per_image'][0]}-{CONFIG['objects_per_image'][1]}")

    output_dir = Path(CONFIG["output_dir"])
    images_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    out_w, out_h = CONFIG["output_size"]

    for i in range(CONFIG["composite_count"]):
        bg = random.choice(backgrounds).copy()
        bg = cv2.resize(bg, (out_w, out_h))

        num_objects = random.randint(*CONFIG["objects_per_image"])
        selected_crops = random.choices(grok_crops, k=num_objects)

        placed_objects = []
        for crop_info in selected_crops:
            crop = crop_info["image"]
            scale = random.uniform(*CONFIG["composite_scale_range"])

            new_h = int(crop.shape[0] * scale)
            new_w = int(crop.shape[1] * scale)

            # Enforce max 1/4 of image area
            max_area = (out_w * out_h) * 0.25
            if new_h * new_w > max_area:
                shrink = (max_area / (new_h * new_w)) ** 0.5
                new_h = int(new_h * shrink)
                new_w = int(new_w * shrink)
                scale = scale * shrink

            max_y = max(0, out_h - new_h)
            max_x = max(0, out_w - new_w)
            pos_y = random.randint(0, max_y) if max_y > 0 else 0
            pos_x = random.randint(0, max_x) if max_x > 0 else 0

            bg, bbox = paste_crop_on_background(bg, crop, (pos_y, pos_x), scale)
            if bbox is not None:
                placed_objects.append({"name": crop_info["class"], "bbox": list(bbox)})

        if not placed_objects:
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"composite_fire_{timestamp}_{i:04d}.jpg"
        img_path = str(images_dir / filename)
        xml_path = str(ann_dir / f"composite_fire_{timestamp}_{i:04d}.xml")

        cv2.imwrite(img_path, bg)
        save_composite_voc_xml(img_path, bg.shape, placed_objects, xml_path)

        obj_summary = ", ".join(f"{o['name']}" for o in placed_objects)
        print(f"  ✅ [{i+1}/{CONFIG['composite_count']}] {filename} ({len(placed_objects)} objects: {obj_summary})")
        generated += 1

    print(f"\n  Composite generation done: {generated} images created")
    return generated


# ===================== MAIN GENERATION FUNCTION =====================

def generate_dataset():
    """Generate full dataset of fire/smoke images."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # COMPOSITE MODE: Paste fire/smoke crops onto your backgrounds
    # ================================================================
    composite_count = 0
    if CONFIG.get("composite_mode", False):
        composite_count = generate_composite_images()

    # ================================================================
    # GROK API GENERATION — Security surveillance fire & smoke images
    # ================================================================
    prompts_file = output_dir / "prompts_log.txt"

    print(f"\n🔥 Grok Security Surveillance Fire & Smoke Dataset Generator")
    print(f"{'='*60}")
    print(f"API: xAI Grok ({CONFIG['model']})")
    print(f"Output: {output_dir}")
    print(f"Target: {CONFIG['num_images']} images (1280x720 surveillance perspective)")
    print(f"Distribution: ~{int(CONFIG['smoke_only_probability']*100)}% smoke-only / ~{int((1-CONFIG['smoke_only_probability'])*100)}% fire")
    print(f"{'='*60}")

    print(f"\n🔌 Testing xAI API connection...")
    try:
        client = setup_xai_client()
        print(f"✅ API connected successfully")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return

    success_count = 0
    fail_count = 0

    specific_prompts = generate_specific_scenario_prompts()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_file, 'w') as pf:
        for i in range(CONFIG["num_images"]):
            print(f"\n[{i+1}/{CONFIG['num_images']}] Generating image...")

            if i < len(specific_prompts):
                prompt = specific_prompts[i]
                is_smoke = ("no fire" in prompt.lower() or "no flames" in prompt.lower() or "no visible flame" in prompt.lower())
                scene_type = "surveillance_smoke_specific" if is_smoke else "surveillance_fire_specific"
            else:
                rand = random.random()
                if rand < CONFIG["smoke_only_probability"]:
                    prompt = generate_smoke_prompt()
                    scene_type = "surveillance_smoke"
                else:
                    prompt = generate_fire_prompt()
                    scene_type = "surveillance_fire"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grok_{scene_type}_{timestamp}_{i:04d}.png"
            output_path = images_dir / filename

            pf.write(f"\n{'='*60}\n")
            pf.write(f"Image {i+1}: {filename}\n")
            pf.write(f"Type: {scene_type}\n")
            pf.write(f"Prompt:\n{prompt}\n")

            success = generate_image_with_grok(prompt, str(output_path))

            if success:
                success_count += 1
                print(f"  ✅ Saved: {filename}")
            else:
                fail_count += 1
                print(f"  ❌ Failed to generate image {i+1}")

            if i < CONFIG["num_images"] - 1:
                time.sleep(CONFIG["delay_between_requests"])

    print(f"\n{'='*60}")
    print(f"🎉 Grok generation complete!")
    print(f"✅ Successful: {success_count}/{CONFIG['num_images']}")
    print(f"❌ Failed: {fail_count}/{CONFIG['num_images']}")

    # Composite mode (disabled by default, enable in CONFIG)
    composite_count = 0
    if CONFIG.get("composite_mode", False):
        composite_count = generate_composite_images()

    print(f"\n{'='*60}")
    print(f"📊 DATASET SUMMARY:")
    print(f"  Grok-generated: {success_count}")
    print(f"  Composite: {composite_count}")
    print(f"  Total: {success_count + composite_count}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")


def preview_prompts(num_samples: int = 10):
    """Preview sample prompts without generating images."""
    print(f"\n🔍 Sample Prompts Preview (Security Surveillance Fire & Smoke Detection)\n{'='*60}\n")
    
    print("SURVEILLANCE FIRE PROMPTS (FIRE AS PRIMARY FOCUS):")
    print("-" * 60)
    for i in range(num_samples // 2):
        print(f"\n{i+1}. {generate_fire_prompt()}")
    
    print(f"\n\nSURVEILLANCE SMOKE-ONLY PROMPTS (NO FIRE VISIBLE):")
    print("-" * 60)
    for i in range(num_samples // 2):
        print(f"\n{i+1}. {generate_smoke_prompt()}")
    
    print(f"\n\nSPECIFIC HIGH-PRIORITY SCENARIOS (Surveillance Perspective):")
    print("-" * 60)
    for i, prompt in enumerate(generate_specific_scenario_prompts(), 1):
        print(f"\n{i}. {prompt}")
    
    print(f"\n{'='*60}")
    print("\n🎯 Image Generation Focus:")
    print("   ✅ FIRE images: Bright visible flames (primary subject)")
    print("   ✅ FIRE with smoke: Compact localized smoke (30% of fire images)")
    print("   ✅ SMOKE-ONLY images: Compact smoke without fire (25% of all images)")
    print("   ✅ All smoke: Localized in tight columns - NOT spread out")
    print("   ✅ Easy to annotate - smoke in tight vertical columns/plumes")
    print("   ❌ NO room-filling smoke or spread-out haze")
    print("   ✅ Indoor (60%) + Outdoor (40%) locations")
    print("   ✅ Updated location variety for diversity")
    print("   ✅ Security camera perspectives (CCTV angles)")
    print("\n📹 Camera Perspectives:")
    print("   - Ceiling-mounted CCTV angles")
    print("   - Wall-mounted surveillance views")
    print("   - Corner security camera perspectives")
    print("   - Wide-angle security lens distortion")
    print("   - Overhead monitoring camera views")
    print("\n🏢 Location Types:")
    print("   - Indoor: Factories, data centers, retail, offices, etc.")
    print("   - Outdoor: Parking lots, loading docks, rooftops, etc.")
    print("\n💨 Smoke Characteristics (ALL smoke images):")
    print("   - Compact vertical columns")
    print("   - Localized in contained area")
    print("   - Tight plumes - easy bounding box")
    print("   - Background remains clearly visible")
    print("   - NO spread-out or room-filling smoke")
    print(f"{'='*60}")


def export_prompts_only():
    """Export all prompts to a file without generating images."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_file = output_dir / "prompts_for_other_models.txt"
    
    print(f"\n📝 Exporting prompts for use with other image generation services...")
    
    with open(prompts_file, 'w') as f:
        f.write("SECURITY SURVEILLANCE FIRE & SMOKE DETECTION IMAGE PROMPTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Generated by Grok AI (xAI)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Focus: Fire Detection + Smoke Detection - Security Surveillance Perspectives\n")
        f.write("Fire: Bright visible flames (primary subject)\n")
        f.write("Smoke: Compact localized smoke (easy to annotate)\n")
        f.write("Locations: Indoor (60%) + Outdoor (40%)\n")
        f.write("Camera: Ceiling/wall-mounted CCTV angles\n")
        f.write("=" * 80 + "\n\n")
        
        # Specific scenarios
        f.write("\nSPECIFIC HIGH-PRIORITY SURVEILLANCE FIRE SCENARIOS:\n")
        f.write("-" * 80 + "\n")
        for i, prompt in enumerate(generate_specific_scenario_prompts(), 1):
            f.write(f"\n{i}. {prompt}\n")
        
        # Random fire prompts
        f.write("\n\nRANDOM SURVEILLANCE FIRE SCENARIOS (40 variations):\n")
        f.write("-" * 80 + "\n")
        for i in range(40):
            prompt = generate_fire_prompt()
            f.write(f"\n{i+1}. {prompt}\n")
        
        # Random smoke-only prompts
        f.write("\n\nRANDOM SURVEILLANCE SMOKE-ONLY SCENARIOS (30 variations):\n")
        f.write("-" * 80 + "\n")
        for i in range(30):
            prompt = generate_smoke_prompt()
            f.write(f"\n{i+1}. {prompt}\n")
    
    print(f"✅ Prompts exported to: {prompts_file}")
    print(f"💡 Use these prompts with DALL-E, Midjourney, Stable Diffusion, etc.")


# ===================== CLI =====================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   Grok AI Fire & Smoke Detection Image Generator (xAI)    ║
    ║   Security Surveillance Camera Perspective                 ║
    ║   Fire + Smoke Detection - CCTV/Security Camera Angles     ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--preview":
            preview_prompts()
        elif command == "--export-prompts":
            export_prompts_only()
        elif command == "--help":
            print("""
Usage:
    python generate-images-grok.py              Generate fire & smoke images (surveillance perspective)
    python generate-images-grok.py --preview    Preview sample prompts
    python generate-images-grok.py --export-prompts    Export prompts to file
    python generate-images-grok.py --help       Show this help

Current Configuration:
    - Fire images: ~75% (with optional compact smoke)
    - Smoke-only images: ~25% (no fire visible)
    - All smoke: Compact & localized (easy to annotate)
    - Locations: Indoor (60%) + Outdoor (40%)
    - Perspective: Security surveillance cameras
    - Camera angles: Ceiling/wall-mounted CCTV views
    - Style: Security footage, wide-angle lens

Complete Workflow:
    1. Generate images:
       python generate-images-grok.py
    
    2. Auto-annotate with your YOLO model:
       python annotate-generated-images.py
    
    3. Get Pascal VOC XML annotations automatically!
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information")
    else:
        generate_dataset()

