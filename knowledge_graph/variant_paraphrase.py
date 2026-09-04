# -*- coding: utf-8 -*-
"""Paraphrase variant: the wording axis of the perturbation study.

Every sentence is rewritten in different words and a different construction
while the facts are preserved exactly: the same mobility, the same symptoms,
the same locations and colours, and the same contrastive targets. The variant
tests whether the prototypes depend on one annotator's phrasing rather than on
the agronomic content the schema requires.
"""

PARAPHRASE = {

"N": (
    "Because nitrogen translocates readily within the plant, the oldest foliage shows damage first "
    "while nitrogen is withdrawn towards developing tissue. "
    "1. Overall appearance: The characteristic sign is a whole-blade fading to a washed-out yellow-green. "
    "What sets it apart from iron or magnesium shortage is that the venation yellows along with the lamina "
    "instead of staying green. "
    "2. Evenness: The fading is normally even across the blade, although in the earliest phase scattered "
    "asymmetric pale zones may precede full coverage. "
    "3. Tissue death: With advancing senescence, broad brown dead areas emerge, usually initiating at the "
    "apex or along the edges, and the leaf is shed early. "
    "4. Dimensions: Growth is suppressed, so blades end up both smaller and thinner than normal."
),

"P": (
    "Phosphorus moves freely in the plant, so the symptoms are seen on mature foliage. "
    "1. Pigmentation: What distinguishes this disorder from every other one is the appearance of purple-red, "
    "violet or bronzed patches produced by anthocyanin build-up in the lamina. "
    "2. Underlying tone: Ahead of the reddening, older blades commonly take on a muted blue-tinged dark green, "
    "which is not the washed-out yellow that nitrogen shortage produces. "
    "3. Distribution: The red-violet colour typically emerges in discrete patches that later coalesce with "
    "yellowed zones, giving the blade a mottled, autumnal look. "
    "4. Tissue death: In later stages sizeable brown dead patches appear inside the discoloured zones, "
    "concentrated at the apex and along the edges."
),

"K": (
    "Potassium relocates easily, so damage begins on mature foliage and typically at the apex. "
    "1. Overall appearance: The diagnostic feature is edge death, with the blade margins looking burnt, "
    "brown and desiccated, as though scorched. "
    "2. Zonation: A clear three-band sequence is normally visible, running from the dead brown rim through "
    "a thin yellowed chlorotic band and into the still-green interior. "
    "3. Spread: Death generally begins at the leaf apex and travels along the margins, while the midrib "
    "region and inner lamina stay green and functional for a considerable period. "
    "4. Feel: The dead tissue is dry and snaps easily and tends to roll upwards, which differs from the "
    "soft tissue death produced by other shortages."
),

"Ca": (
    "Calcium does not travel in the phloem, so damage is restricted to newly formed tissue and growing points. "
    "1. Shape: The hallmark is a physically altered blade edge that becomes markedly undulating, frilled or "
    "scalloped. "
    "2. Distortion: Blades frequently end up hooked or twisted towards the apex because cells expand unevenly. "
    "3. Pigmentation: In place of the delicate venation pattern typical of iron shortage, calcium shortage "
    "produces broad, ill-defined pale patches beginning at the edges or between veins, which often bronze or "
    "die back. "
    "4. Tissue death: Once cell walls give way, small brown dead specks or perforations open up inside the "
    "pale zones."
),

"Mg": (
    "Magnesium relocates within the plant, so mature foliage is affected first. "
    "1. Overall appearance: The signature pattern is chlorosis between the veins in which the midrib and the "
    "principal side veins stay flanked by broad green margins, producing a herringbone effect. "
    "2. Comparison: Set against the delicate venation pattern of iron shortage, the pale zones here are "
    "coarser and the finest veins are not spared. "
    "3. Spread: As the shortage deepens, irregular brown dead specks form inside the pale interveinal zones, "
    "which seldom happens with iron shortage. "
    "4. Feel: Blade shape is largely preserved, but the tissue can feel coarser and more mature than the "
    "tender young foliage affected by iron shortage."
),

"Fe": (
    "Iron is not redistributed within the plant, so only the newest foliage is affected. "
    "1. Overall appearance: The diagnostic feature is a delicate lattice of green venation standing out "
    "against a pale background. In contrast to magnesium shortage, even the finest second- and third-order "
    "veins hold their green colour crisply. "
    "2. Pigmentation: Tissue between the veins shifts from faint green to clear yellow and, when severe, to "
    "an almost ivory or white-yellow tone as chlorophyll is lost. "
    "3. Shape: Set against calcium or boron shortage, the outline of the blade stays essentially normal, "
    "without marked buckling or rippled edges, though blades may be smaller. "
    "4. Spread: The disorder advances from the base towards the apex, and in severe cases the blade bleaches "
    "completely without dying first."
),

"Mn": (
    "Manganese is not redistributed, so young and intermediate-aged foliage carries the symptoms. "
    "1. Overall appearance: The diagnostic feature is interveinal paling with a speckled or stippled look, "
    "commonly likened to a chequerboard of green and yellow flecks, rather than the well-defined bands "
    "produced by magnesium shortage. "
    "2. Placement: Paling tends to begin close to the margins and apex and works inwards, while iron "
    "shortage pales the whole blade evenly. "
    "3. Tissue death: With progression, small isolated brown dead specks form along the principal veins or "
    "inside the pale zones. "
    "4. Feel: The lamina may become slightly puckered or crinkled, though far less markedly than under "
    "boron shortage."
),

"B": (
    "Poor phloem movement means the shortage strikes the growing tip hardest. "
    "1. Shape: New foliage is conspicuously lopsided and malformed, commonly twisted, hooked or otherwise "
    "irregular along the edges. "
    "2. Feel: The lamina turns tough and leathery and snaps readily, with a puckered, wrinkled surface. "
    "3. Pigmentation: Paling is patchy rather than even, showing up as scattered irregular interveinal "
    "blotches ranging from olive to yellow, which is not the delicate venation pattern of iron shortage. "
    "4. Venation: Corky thickening may develop on the underside of the midrib and principal veins, and in "
    "advanced cases dead lesions form at the apex."
),

"Healthy": (
    "An unaffected coffee leaf provides the reference appearance. "
    "1. Pigmentation: The blade is an even, saturated, lustrous dark green, showing full chlorophyll content "
    "and no paling whatsoever. "
    "2. Feel: The surface is even and waxy and reflects light uniformly, without the puckered texture that "
    "boron shortage causes. "
    "3. Shape: The outline is elliptical with even, softly undulating margins typical of the species, and "
    "shows none of the irregular buckling, rolling or hooking associated with calcium or boron shortage. "
    "4. State: The lamina is wholly unmarked, with no dead tissue, spotting or feeding damage."
),
}
