# Step 1: Base image with Python 3.11
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy requirements first (better caching)
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the app
COPY . .

# Step 6: Expose Streamlit default port
EXPOSE 8501

# Step 7: Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
