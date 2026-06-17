# utils/taxonomy.py
# 咖啡叶片病理学描述字典 (CP-KG Text Corpus)
# 参考文献来源:
# [1] Malavolta, E. (2006). Manual de nutrição mineral de plantas.
# [2] Yara Crop Nutrition - Coffee Deficiency Guide.
# [3] IPNI - International Plant Nutrition Institute.
# [4] Wintgens, J. N. (2004). Coffee: Growing, Processing, Sustainable Production.

CLASS_DESCRIPTIONS = {
    # --- 1. Nitrogen (N) ---
    # 关键词: 老叶, 全叶弥漫黄化, 叶脉变黄, 早衰坏死
    "N": (
        "Nitrogen is highly mobile, causing symptoms on older leaves first as N is remobilized to new growth. "
        "1. Visual Pattern: The dominant symptom is 'general chlorosis' where the entire leaf blade turns pale yellow-green. Crucially, unlike Fe or Mg deficiency, the veins also lose their green color and turn yellow. "
        "2. Uniformity: Chlorosis is typically uniform, but early stages may show diffuse, asymmetrical yellowing patches before covering the whole leaf. "
        "3. Necrosis: As deficiency progresses to senescence, large brown necrotic lesions develop, often starting from the tip or margins, leading to premature leaf drop. "
        "4. Size: Leaves are generally smaller and thinner due to stunted growth."
    ),

    # --- 2. Phosphorus (P) ---
    # 关键词: 老叶, 红紫色/青铜色, 花青素积累, 暗绿背景
    "P": (
        "Phosphorus deficiency is mobile, appearing on older leaves. "
        "1. Coloration: The unique symptom is the development of reddish-purple, violet, or bronze blotches on the leaf surface due to anthocyanin accumulation, distinct from any other deficiency. "
        "2. Background: Before turning red, older leaves often exhibit a dull, dark bluish-green color, unlike the pale yellow of N deficiency. "
        "3. Pattern: The red or purple coloration often starts as patches and can merge with yellowing areas, creating a multi-colored 'autumn leaf' appearance. "
        "4. Necrosis: As the deficiency advances, large brown necrotic spots form within the discolored areas, particularly at the tips and margins."
    ),

    # --- 3. Potassium (K) ---
    # 关键词: 老叶, 边缘焦枯, 绿-黄-褐渐变, 叶尖坏死
    "K": (
        "Potassium is highly mobile, so symptoms start on older leaves, often beginning at the tip. "
        "1. Visual Pattern: The defining symptom is marginal necrosis, where the leaf edges appear scorched, brown, and dry, as if burnt by fire. "
        "2. Gradient: There is typically a distinct progression from the necrotic brown edge to a narrow chlorotic yellow halo, and finally to the healthy green center, creating a three-zone effect. "
        "3. Progression: Necrosis usually starts at the leaf tip and advances down the margins, leaving the central midrib and inner tissue green and functional for a long time. "
        "4. Texture: The necrotic tissue is brittle and may curl upward, contrasting with the soft necrosis of other deficiencies."
    ),

    # --- 4. Calcium (Ca) ---
    # 关键词: 新叶, 边缘波浪状, 块状黄化, 钩状畸形
    "Ca": (
        "Calcium is immobile in the phloem, causing symptoms to manifest exclusively on new growth and meristems. "
        "1. Morphology: The defining characteristic is the physical deformation of the leaf margin, which becomes distinctly wavy, ruffled, or scalloped. "
        "2. Deformation: Leaves often exhibit a 'hooked' or distorted shape at the tip due to uneven cell expansion. "
        "3. Coloration: Unlike the fine network of Fe deficiency, Ca deficiency causes diffuse, irregular chlorotic patches that start from the margins or interveinal areas, often turning bronze or necrotic as tissue dies. "
        "4. Necrosis: Small, brown necrotic spots or holes may form within the chlorotic areas as the cell walls collapse."
    ),

    # --- 5. Magnesium (Mg) ---
    # 关键词: 老叶, 鱼骨状黄化, 宽绿带, 内部坏死斑
    "Mg": (
        "Magnesium is mobile, so symptoms appear on older leaves first. "
        "1. Visual Pattern: The hallmark symptom is interveinal chlorosis where the midrib and major lateral veins remain bordered by wide bands of healthy green tissue, creating a distinct 'herringbone' pattern. "
        "2. Contrast: Unlike the fine network of Fe deficiency, the chlorosis in Mg deficiency is blockier and does not affect the finest veins. "
        "3. Progression: As deficiency worsens, irregular brown necrotic spots develop within the chlorotic interveinal areas, which is rare in Fe deficiency. "
        "4. Texture: Leaves generally retain their shape but may feel thicker or older compared to the delicate young leaves of Fe deficiency."
    ),

    # --- 6. Iron (Fe) ---
    # 关键词: 新叶, 精细网状脉, 极度褪绿/发白, 叶形完整
    "Fe": (
        "Iron is immobile, so symptoms appear exclusively on the youngest leaves. "
        "1. Visual Pattern: The defining symptom is a 'fine reticulate network' of green veins against a chlorotic background. Unlike Mg deficiency, even the smallest secondary and tertiary veins remain sharply green. "
        "2. Coloration: The interveinal tissue turns from pale green to distinct yellow, and in severe cases, becomes almost whitish-yellow or ivory due to lack of chlorophyll. "
        "3. Morphology: Unlike Ca or B deficiency, the leaf shape generally remains intact without significant deformation or wavy margins, although the size may be reduced. "
        "4. Progression: Symptoms progress from the base to the tip, and severe deficiency leads to complete bleaching without initial necrosis."
    ),

    # --- 7. Manganese (Mn) ---
    # 关键词: 新叶/中叶, 棋盘格/斑驳, 边缘黄化, 细小坏死
    "Mn": (
        "Manganese is immobile, affecting young to middle-aged leaves. "
        "1. Visual Pattern: The defining symptom is interveinal chlorosis that appears 'mottled' or stippled, often described as a 'checkerboard' pattern of yellow and green spots, unlike the distinct bands of Mg deficiency. "
        "2. Distribution: Chlorosis often starts near the leaf margins and tips and progresses inward, whereas Fe deficiency affects the entire leaf uniformly. "
        "3. Necrosis: Small, scattered brown necrotic spots may develop along the main veins or within the chlorotic areas as the deficiency advances. "
        "4. Texture: The leaf surface may appear slightly rugose or crinkled, but less severely than in Boron deficiency."
    ),

    # --- 8. Boron (B) ---
    # 关键词: 顶端分生组织, 不对称, 皮革质/起皱, 斑驳, 栓化
    "B": (
        "Deficiency critically targets the apical meristem due to low phloem mobility. "
        "1. Morphology: Young leaves exhibit distinct asymmetry and deformation, often appearing twisted, hooked, or misshapen with irregular margins. "
        "2. Texture: The leaf blade develops a 'leathery' and brittle texture with a rugose (crinkled/puckered) surface. "
        "3. Coloration: Chlorosis is not uniform; instead, it manifests as irregular, interveinal mottling (olive-green to yellow blotches) scattered across the leaf, distinct from the fine network of Fe deficiency. "
        "4. Veins: The midrib and main veins may exhibit suberization (corkiness) on the underside, and severe cases lead to necrotic lesions at the leaf tip."
    ),

    # --- 9. Healthy ---
    # 关键词: 深绿光泽, 表面光滑, 规则波浪边, 无瑕疵
    "Healthy": (
        "A healthy coffee leaf serves as the visual baseline. "
        "1. Coloration: The leaf exhibits a uniform, deep, and glossy dark green color, indicating optimal chlorophyll content without any chlorosis. "
        "2. Texture: The surface is smooth and waxy, reflecting light evenly, unlike the rugose texture of B deficiency. "
        "3. Morphology: The leaf shape is elliptical with regular, gently wavy margins (undulate) characteristic of the species, but lacks the irregular distortion, curling, or hooking seen in Ca or B deficiency. "
        "4. Condition: The leaf blade is entirely blemish-free, showing no signs of necrosis, spots, or insect damage."
    )
}

# 类别映射索引 (与代码中的 label_encoder 保持一致)
CLASS_TO_IDX = {
    "B": 0, "Ca": 1, "Fe": 2, "Healthy": 3, "K": 4,
    "Mg": 5, "Mn": 6, "N": 7, "P": 8
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}