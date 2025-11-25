## 🚀 How to Use Now

### Step 1: Train the Model (First Time Only)

Before using the app, you need to train the model:

```bash
# Install dependencies first
pip install -r requirements.txt

# Then train the model
python setup_and_train.py
```

This will take ~5-10 minutes and creates:

- `models/xgb_adr_model.pkl`
- `models/shap_explainer.pkl`
- Reports and visualizations

### Step 2: Launch the Streamlit App

```bash
streamlit run src/app.py
```

### Step 3: Test the App

1. **Enter patient details** on the "Patient Entry" page
   - Fill in age, gender, comorbidities
   - Select medications (optional)
   - Enter lab values
2. **Click "🔍 Predict ADR Risk"**
   - You'll see: "✅ Patient data saved! Switching to Prediction Results..."
   - App automatically navigates to the Results page
3. **View results** on the "Prediction Results" page
   - Risk score (0-100%)
   - Risk category (Low/Moderate/High)
   - Top contributing factors
   - Export options

## 📋 What to Expect

### Before Training

If you try to use the app before training:

- ❌ Error: "Model not found"
- 💡 Message: "Please run training first: `python src/train_xgb.py`"

### After Training

- ✅ Click "Predict ADR Risk" button
- ✅ See success message
- ✅ Auto-navigate to "Prediction Results" page
- ✅ View risk score, gauge, and explanations
- ✅ Can navigate to other pages (Explainability, Performance, etc.)

## 🔧 Troubleshooting

### "Model not found" error

```bash
# Solution: Train the model
python setup_and_train.py
```

### Button clicks but nothing happens

- Check the Streamlit terminal for errors
- Make sure you're on "Patient Entry" page
- Try refreshing the browser (Ctrl/Cmd + R)

### Page doesn't switch automatically

- Look for the success message: "✅ Patient data saved! Switching to Prediction Results..."
- If you see it, wait 1-2 seconds for the page to reload
- If not, manually click "2️⃣ Prediction Results" in the sidebar

## 📝 Complete Workflow

```
1. Patient Entry Page
   ↓ (Enter data & click "Predict ADR Risk")
   ↓
2. [Auto-navigates to Prediction Results]
   ↓
3. View risk score, gauge, contributors
   ↓
4. Optionally: Navigate to Explainability page
   ↓
5. View SHAP explanations (global/local)
   ↓
6. Optionally: Check Model Performance page
   ↓
7. View metrics, fairness audit
```

## 🎯 Quick Test

**Try this after training:**

1. Launch app: `streamlit run src/app.py`
2. Stay on "Patient Entry" page
3. Use default values (age=65, gender=M, etc.)
4. Click "🔍 Predict ADR Risk"
5. Watch for success message
6. Page auto-switches to show results
7. You should see a risk gauge and score!

---

**Status: FIXED & READY TO USE** ✅

The app now provides a smooth user experience with automatic navigation!
