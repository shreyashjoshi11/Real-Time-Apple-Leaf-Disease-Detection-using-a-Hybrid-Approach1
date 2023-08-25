1. Setup for Python:
Install Python (Setup instructions)

Install Python packages

pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt

Install Tensorflow Serving 

2.Setup for ReactJS
Install Nodejs 
Install NPM 
Install dependencies
cd frontend
npm install --from-lock-json
npm audit fix
Copy .env.example as .env.

Change API url in .env.

3. Training the Model:
Download the data from kaggle.
Only keep folders related to Apples.
Run Jupyter Notebook in Browser.
jupyter notebook
Open Hybrid Model.ipynyb in Jupyter Notebook.
Run all the Cells one by one.
Copy the model generated and save it in the models1 folder.

4.Run the Main1.py via uvicorn use 
uvicorn main1:app --reload

5.Get inside the Frontend Folder and open GIT Bash to run npm run start , this will launch the frontend.
