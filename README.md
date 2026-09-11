# SimRORC - Simulated Recuperated ORC

Design-mode simulator for a recuperated organic Rankine cycle on a liquid geothermal
or waste-heat source. By **Md Faisal Karim** - [LinkedIn](https://www.linkedin.com/in/md-faisal-karim).

Run locally:

    pip install -r requirements.txt
    streamlit run SimRORC_app.py

## Publish for free (GitHub + Streamlit Community Cloud)

1. Create a GitHub repository (public or private) and upload everything in this folder
   (`SimRORC_app.py`, `ORC_recuperated.py`, `Turbine.py`, `Pump.py`, `HEX.py`,
   `AirCooledCondenser.py`, `requirements.txt`, `.streamlit/config.toml`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, pick the
   repository, branch `main`, main file `SimRORC_app.py`, and **Deploy**.
3. After a few minutes the app is live at `https://<name>.streamlit.app`.
   Every later push to GitHub redeploys it automatically.

GitHub only stores the code (GitHub Pages cannot run Python); Streamlit Community Cloud
is what runs it.

## Show it under your own domain (DreamHost shared hosting)

1. Open `dreamhost/index.html`, replace `YOUR-APP` with the name from step 3 above.
2. In the DreamHost panel, either create a folder (e.g. `simrorc`) inside your existing
   website and upload `index.html` there  ->  `https://yourdomain.com/simrorc/`,
   or add a subdomain (e.g. `simrorc.yourdomain.com`) and upload `index.html` as its home page.
3. Done: the page frames the running app; the address bar shows your domain.

Notes: free Streamlit apps sleep after inactivity and wake in about a minute on the first
visit; an optimisation takes roughly 15-40 s on the shared cloud CPU.
