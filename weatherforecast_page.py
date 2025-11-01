# weatherforecast_page.py
import streamlit as st
import pandas as pd


from weather import get_15day_forecast as get_14day_forecast, flatten_forecast, WeatherError


def render():
    st.title("☁️ 14-day Weather Forecast")

    if st.button("← Back to Plant Assistant"):
        st.query_params["view"] = "main"
        st.rerun()

    place = st.text_input("Location / City / lat,lon", "New Delhi, IN")
    go = st.button("Fetch forecast", type="primary")

    if go and place.strip():
        try:
            data = get_14day_forecast(place.strip())
            rows = flatten_forecast(data)
            if not rows:
                st.info("No forecast data returned.")
            else:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except WeatherError as we:
            st.error(str(we))
        except Exception as e:
            st.error(f"Error: {e}")
