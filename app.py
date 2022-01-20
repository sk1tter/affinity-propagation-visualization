import streamlit as st
import matplotlib.pyplot as plt
from affinity_propagation import affinity_prop

from generate_data import generate_data


def validate_preference(preference):
    if preference == "median":
        return True
    try:
        float(preference)
        return True
    except ValueError:
        return False


def main():
    st.set_page_config(
        page_title="Affinity Propagation clustering",
    )

    with st.sidebar.form(key="Data Generator"):
        st.write("## Data Generator")
        data_size = st.slider("Data size", min_value=100, max_value=300)

        data_cluster_size = st.slider(
            "Number of Clusters", min_value=1, max_value=10, value=2
        )

        cluster_std = st.slider(
            "Standard Deviation of Clusters", min_value=0.0, max_value=10.0, value=1.0
        )
        generate_button = st.form_submit_button(label="Generate")

    with st.sidebar.form(key="Affinity Propagation"):
        st.write("## Affinity Propagation Options")
        max_iterations = st.slider(
            "Maximum iterations", min_value=100, max_value=500, value=100
        )
        local_threshold = st.slider(
            "Local threshold", min_value=10, max_value=30, value=10
        )
        damping_factor = st.slider(
            "Damping factor", min_value=0.1, max_value=1.0, value=0.7
        )
        preference = st.text_input("Preference", value="median")
        train_button = st.form_submit_button(label="Train")

    st.title(
        "Affinity Propagation clustering",
    )

    if "fig" not in st.session_state:
        st.session_state.fig, st.session_state.ax = plt.subplots(figsize=(14, 12))
        st.session_state.ax.axes.xaxis.set_visible(False)
        st.session_state.ax.axes.yaxis.set_visible(False)
        st.session_state.data_plot = st.pyplot(st.session_state.fig)
        st.session_state.res_box = st.empty()
    else:
        st.session_state.ax.clear()
        st.session_state.res_box.empty()

    if generate_button:
        st.session_state.sample_data = generate_data(
            data_cluster_size, cluster_std, data_size
        )[0]

        st.session_state.ax.plot(
            st.session_state.sample_data[:, 0],
            st.session_state.sample_data[:, 1],
            ".",
            alpha=0.5,
        )
        st.session_state.data_plot.pyplot(st.session_state.fig)

    if train_button:
        if not validate_preference(preference):
            st.error("Preference must be either 'median' or a numeric value.")
            return

        if "sample_data" not in st.session_state:
            st.error("You must generate data first.")
            return

        affinity_prop(
            st.session_state.sample_data,
            maxiter=max_iterations,
            damping_factor=damping_factor,
            preference=preference,
            local_thresh=local_threshold,
            data_plot=st.session_state.data_plot,
            fig=st.session_state.fig,
            ax=st.session_state.ax,
            c=st.session_state.res_box,
        )


if __name__ == "__main__":
    main()
