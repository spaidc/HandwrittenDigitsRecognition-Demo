# Huong Dan Chay Demo

File nay ghi lai cac lenh can dung de mo lai virtual environment va chay app Streamlit.

## 1. Mo Terminal Dung Thu Muc Project

Mo Command Prompt hoac PowerShell, sau do chay:

```powershell
cd C:\Users\Admin\Documents\Codes\AI\DemoUI
```

## 2. Kich Hoat Lai Virtual Environment

Neu dang dung Command Prompt:

```cmd
.\.venv\Scripts\activate.bat
```

Neu dang dung PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Sau khi activate thanh cong, dau dong lenh se co dang:

```text
(.venv) C:\Users\Admin\Documents\Codes\AI\DemoUI>
```

Neu khong thay `(.venv)`, nghia la ban chua vao dung moi truong ao.

## 3. Kiem Tra Python Dang Dung

Chay:

```powershell
python --version
python -m pip --version
```

Ket qua nen la Python 3.10+ hoac 3.11+. Khong nen la Python 3.8.

Luon uu tien dung:

```powershell
python -m pip ...
```

thay vi:

```powershell
pip ...
```

de tranh pip tro nham sang Python khac.

## 4. Cai Lai Dependencies Khi Can

Chi can chay buoc nay khi moi clone repo, moi tao `.venv`, hoac `requirements.txt` vua thay doi:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Chay Demo

```powershell
python -m streamlit run app.py
```

Sau do mo trinh duyet tai:

```text
http://localhost:8501
```

## 6. Loi Thuong Gap

### Van bi dung Python 3.8

Neu thay duong dan co `Python38`, ban chua activate dung `.venv`.

Hay dong terminal, mo lai terminal moi, vao dung thu muc project va chay lai:

```powershell
cd C:\Users\Admin\Documents\Codes\AI\DemoUI
.\.venv\Scripts\Activate.ps1
python --version
```

### PowerShell chan Activate.ps1

Chay lenh nay trong cung terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Thieu streamlit_drawable_canvas

Chay lai:

```powershell
python -m pip install -r requirements.txt
```

### Thieu TensorFlow

Kiem tra truoc:

```powershell
python --version
```

Neu dang la Python 3.8, tao lai venv bang Python 3.11. Neu da la Python 3.11, chay:

```powershell
python -m pip install -r requirements.txt
```

## 7. Tao Lai Venv Neu Bi Loi

Chi dung khi `.venv` cu bi loi hoac bi tao sai Python.

PowerShell:

```powershell
cd C:\Users\Admin\Documents\Codes\AI\DemoUI
Remove-Item .venv -Recurse -Force
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Command Prompt:

```cmd
cd C:\Users\Admin\Documents\Codes\AI\DemoUI
rmdir /s /q .venv
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
