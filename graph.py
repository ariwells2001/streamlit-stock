import streamlit as st
import matplotlib.pyplot as plt
from bokeh.plotting import figure

years = [1950,1960,1970,1980,1990,2000,2010]
gdp= [300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]

fig,ax = plt.subplots(figsize=(15,5))
ax.plot(years,gdp, color='green', marker='o',linestyle='solid')
#ax.plot(years,gdp*1, color='red', marker='o',linestyle='solid')

ax.set_title("Billions of $")
ax.set_xlabel("years")
ax.set_ylabel("$")
st.pyplot(fig)

fig,axs = plt.subplots(2,2)
axs[0,0].plot(years,gdp, color='green', marker='o',linestyle='solid')
axs[0,1].plot(years,gdp, color='red', marker='o',linestyle='solid')
axs[1,0].plot(years,gdp, color='yellow', marker='o',linestyle='solid')
axs[1,1].plot(years,gdp, color='black', marker='o',linestyle='solid')
st.pyplot(fig)

x = [1,2,3,4,5]
y = [6,7,2,4,5]

p = figure(
    title = 'simple line example',
    x_axis_label='x',
    y_axis_label='y'
)


p.line(x,y, legend_label='Trend',line_width=2)
st.bokeh_chart(p,use_container_width=True)
