import marimo._code_mode as cm

cell_display_two_parts = '''# Hien thi ket qua chia thanh 2 phan ro ret: mix-dataset-v2 va mix-frcrn
_df_no = df_overall[df_overall["condition"] == "No FRCRN"].drop(columns=["condition"])
_df_fr = df_overall[df_overall["condition"] == "Mix + FRCRN"].drop(columns=["condition"])

tab1_content = mo.vstack([
    mo.md("### PHAN 1: TAP DU LIEU `mix-dataset-v2` (No FRCRN - RAW MIXTURE)"),
    mo.md("""
    - **Nguon du lieu:** `/datasets/mix-dataset-v2/eval/mixture/*.wav` (3,064 mau clean)
    - **Dac diem:** Am thanh hon hop truc tiep giua tieng noi va tap am (tu -5 dB den 20 dB).
    - **Xu ly:** Khong qua FRCRN, khong ap dung RMS norm ngoai (giu nguyen bien do goc).
    """),
    mo.ui.table(_df_no, selection=None),
])

tab2_content = mo.vstack([
    mo.md("### PHAN 2: TAP DU LIEU `mix-frcrn` (Mix + FRCRN - MASKED NOISE)"),
    mo.md("""
    - **Nguon du lieu:** `/datasets/mix-frcrn/validation.npy` (shape `(3064, 64000)`)
    - **Dac diem:** Song am nhieu sau khi loc tieng noi bang mat na FRCRN.
    - **Xu ly:** Chuan hoa nang luong RMS muc tieu $\\\\sigma_0 = 0.07925, \\\\epsilon = 10^{-8}$.
    """),
    mo.ui.table(_df_fr, selection=None),
])

mo.ui.tabs({
    "Phan 1: mix-dataset-v2 (No FRCRN)": tab1_content,
    "Phan 2: mix-frcrn (Mix + FRCRN)": tab2_content,
})
'''

async with cm.get_context() as ctx:
    ctx.edit_cell("ZVGD", code=cell_display_two_parts, hide_code=False)
    ctx.run_cell("ZVGD")

print("Successfully updated cell ZVGD into 2 distinct parts with mo.ui.tabs!")
