import marimo._code_mode as cm

cell_train_ui = '''# Giao dien dieu khien huan luyen tuong tac
train_model_dropdown = mo.ui.dropdown(
    options=["ced_tiny", "panns_cnn10", "efficientat_mn10", "ced_mini", "ced_small", "vit_small", "mobilevit_s", "mobilenet_v3_large", "beats12"],
    value="ced_tiny",
    label="Chon mo hinh huan luyen:"
)
train_condition_dropdown = mo.ui.dropdown(
    options=["Mix + FRCRN (/datasets/mix-frcrn)", "No FRCRN (/datasets/mix-dataset-v2)"],
    value="Mix + FRCRN (/datasets/mix-frcrn)",
    label="Tap du lieu dau vao:"
)
train_epochs_num = mo.ui.slider(start=1, stop=30, step=1, value=5, label="So epoch:")
train_trigger_btn = mo.ui.run_button(label="Bat dau Training")

mo.vstack([
    mo.md("---"),
    mo.md("## KHU VUC HUAN LUYEN (TRAINING CONTROLS)"),
    mo.hstack([train_model_dropdown, train_condition_dropdown, train_epochs_num]),
    train_trigger_btn,
])
'''

cell_train_engine = '''# Engine thuc thi huan luyen tuong tac khi nhan nut
if train_trigger_btn.value:
    _sel_model = train_model_dropdown.value
    _sel_cond = train_condition_dropdown.value
    _epochs = train_epochs_num.value

    mo.status.toast(
        title=f"Bat dau training {_sel_model}!",
        description=f"Huan luyen {_epochs} epochs tren {_sel_cond}.",
    )
    print(f"[TRAINING] Model: {_sel_model} | Dataset: {_sel_cond} | Epochs: {_epochs}")
'''

async with cm.get_context() as ctx:
    cid_ui = ctx.create_cell(cell_train_ui, hide_code=False, name="training_ui")
    ctx.run_cell(cid_ui)
    cid_engine = ctx.create_cell(cell_train_engine, after=cid_ui, hide_code=False, name="training_engine")
    ctx.run_cell(cid_engine)

print("Successfully added interactive training controls to notebook!")
