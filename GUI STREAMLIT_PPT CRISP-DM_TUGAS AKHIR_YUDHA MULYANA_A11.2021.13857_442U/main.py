import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from cleandata import clean_data

@st.cache_data
def read_data():
    df = pd.read_csv("Resign.csv")
    df = df[['MASA KERJA',"PT","STATUS KARYAWAN", "JABATAN","LEVEL","DEPARTEMENT","BISNIS",
             "LOKASI","JENIS TURNOVER","ALASAN RESIGN","PLACEMENT","PAKLARING"]]#mengambil kolom yang akan digunakan
    return df#mengembalikan nilai df

def preproses(df):#digunakan untuk mengubah variabel kategorikal menjadi representasi numerik yang dapat digunakan dalam model pemrosesan data.
    df_preproses = pd.get_dummies(df)
    return df_preproses#mengembalikan nilai df_preproses

# melakukan proses Scaling digunakan untuk mengubah variabel-variabel numerik dalam skala yang sama, 
# sehingga memudahkan proses analisis dan perbandingan antar variabel.
def scale_data(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_preproses)
    return scaled_features #mengembalikan nilai scaled_features

#PCA digunakan untuk mengurangi dimensi data sehingga dapat ditampilkan dalam ruang dua dimensi. Selain itu, 
# fungsi ini juga melakukan clustering menggunakan algoritma K-means pada data hasil reduksi dimensi. 
# Hasil clustering ditambahkan ke dataframe utama dan dataframe hasil reduksi dimensi.
def data_PCA(df2,scaled_features,n_clus):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    principal_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
    kmeans = KMeans(n_clusters=n_clus, random_state=42)
    kmeans.fit(principal_df)
    principal_df['Cluster'] = kmeans.labels_
    df2['Cluster'] = kmeans.labels_
    return kmeans.labels_, principal_df
    
    
#Selanjutnya, dataset dibaca menggunakan fungsi read_data() dan hasilnya dimasukkan ke dalam dataframe df.
# Data tersebut kemudian dibersihkan menggunakan fungsi clean_data() yang tidak terlihat dalam kode yang diberikan.
# Setelah itu, dilakukan proses preprocessing pada data menggunakan fungsi preproses()
df = read_data()
df_bersih = clean_data(df)
df = df_bersih.copy()
df_preproses = preproses(df)
#Data yang telah di-preprocessing akan di-scaling menggunakan fungsi scale_data().
scaled_features = scale_data(df_preproses)





# Menambahkan label klaster ke dalam DataFrame hasil PCA
st.title('Clustering K-Means terhadap data Resign')
st.write('Data yang digunakan adalah data resign dari perusahaan X. Data tersebut berisi informasi mengenai masa kerja, PT, status karyawan, jabatan, level, departement, bisnis, lokasi, jenis turnover, alasan resign, placement, dan paklaring. Data tersebut kemudian dibersihkan menggunakan fungsi clean_data() yang tidak terlihat dalam kode yang diberikan. Setelah itu, dilakukan proses preprocessing pada data menggunakan fungsi preproses()')

tab1, tab2 = st.tabs(["Clustering", "Visualization"])

with tab1:
    n_cluster = st.slider('Jumlah cluster', 1,10 ,3)#Digunakan untuk memilih jumlah cluster dengan mengatur slider dengan rentang 1 hingga 10 dan nilai default 3.
    #Digunakan untuk memilih jumlah standard deviasi dengan pilihan 1 hingga 5 dan nilai default 2.
    std_error = st.selectbox("Berapa Jumlah standard deviasi",(1,2,3,4,5),index=1)
    #Digunakan untuk memilih apakah ingin menampilkan confidence level error bar.
    error_bar = st.checkbox("Apakah tambahkan confidence level error bar")
    #fungsi data_PCA() akan dipanggil dengan menggunakan nilai-nilai parameter yang sudah diatur. 
    
    # Hasil dari fungsi tersebut akan disimpan dalam variabel labels dan pca_data.
    labels, pca_data = data_PCA(df_preproses,scaled_features,n_cluster)
    
    with st.container():#st.container() untuk menampilkan plot dan tabel. 
        #Di dalam kontainer ini, dilakukan visualisasi clustering dengan menggunakan scatter plot.
        x_axis = 'PCA1'
        y_axis = 'PCA2'
        hue = 'Cluster'

        # Define a brighter color palette
        cluster_colors = sns.color_palette('bright', n_colors=n_cluster)

        # Scatter plot Digunakan sns.scatterplot() untuk membuat scatter plot dengan sumbu x dan y dari PCA data yang dihasilkan, 
        # dan warna plot akan diatur berdasarkan cluster.
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_axis, y=y_axis, hue=hue, data=pca_data, ax=ax, palette=cluster_colors)
        #ax.set_xlabel(x_axis) dan ax.set_ylabel(y_axis): Mengatur label sumbu x dan y pada scatter plot.
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        # Menghitung rata-rata dan standar deviasi untuk setiap cluster berdasarkan kolom PCA1 dan PCA2 pada dataframe pca_data.
        cluster_stats = pca_data.groupby('Cluster')[['PCA1', 'PCA2']].agg(['mean', 'std'])

        # Membuat checkbox pada aplikasi Streamlit dengan label "Show Subplots per Cluster" 
        # untuk memilih apakah ingin menampilkan subplot per cluster atau tidak.
        checkbox = st.checkbox("Show Subplots per Cluster/individual cluster")
        if checkbox is not True:# Jika checkbox tidak dicentang, 
            #maka akan dilakukan perulangan untuk setiap cluster menggunakan for cluster_label in range(n_cluster).
                for cluster_label in range(n_cluster):
                    #cluster_mean dan cluster_std untuk Mengambil nilai rata-rata dan standar deviasi untuk cluster tertentu dari cluster_stats.
                    cluster_mean = cluster_stats.loc[cluster_label, ('PCA1', 'mean')], cluster_stats.loc[cluster_label, ('PCA2', 'mean')]
                    cluster_std = cluster_stats.loc[cluster_label, ('PCA1', 'std')], cluster_stats.loc[cluster_label, ('PCA2', 'std')]

                    # Menghitung matriks kovarians untuk kolom PCA1 dan PCA2 dari cluster tertentu.
                    cov_matrix = pca_data.loc[pca_data['Cluster'] == cluster_label, ['PCA1', 'PCA2']].cov()
                    cov_ellipse = cov_matrix.values

                    # CMenghitung eigenvalue dan eigenvector dari matriks kovarians untuk menentukan sudut rotasi ellips.
                    eigenvalues, eigenvectors = np.linalg.eig(cov_ellipse)
                    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
                    
                    #Jika checkbox "Apakah tambahkan confidence level error bar" dicentang, 
                    if error_bar:
                    #maka akan membuat confidence ellipse menggunakan Ellipse() berdasarkan standar deviasi. 
                    # Ukuran ellips dapat disesuaikan dengan ellipse_width dan ellipse_height yang dihitung berdasarkan std_error dan cluster_std. 
                        ellipse_width = std_error * cluster_std[0]
                        ellipse_height = std_error * cluster_std[1]
                        confidence_ellipse = Ellipse(xy=cluster_mean, width=2 * ellipse_width, height=2 * ellipse_height,
                                                    angle=angle, edgecolor='green', facecolor=sns.color_palette('bright')[cluster_label],alpha=0.2)

                        # Confidence ellipse ditambahkan ke scatter plot menggunakan ax.add_patch().
                        ax.add_patch(confidence_ellipse)

                # Menambahkan legenda untuk penjelasan warna cluster di pojok kanan bawah plot.
                legend = ax.legend(title='Cluster', loc='lower right')
                for i, text in enumerate(legend.get_texts()):#Mengubah warna teks pada legenda sesuai dengan warna cluster menggunakan text.set_color().
                    text.set_color(cluster_colors[i])

                # Menampilkan scatter plot menggunakan st.pyplot() pada aplikasi Streamlit.
                st.pyplot(fig)
                
                
                #Jika checkbox "Show Subplots per Cluster" dicentang dan jumlah cluster lebih dari 1,
        elif checkbox == True and n_cluster > 1:
            # maka akan dibuat subplot menggunakan plt.subplots(nrows=1, ncols=n_cluster, figsize=(10, 4)) 
            fig, axes = plt.subplots(nrows=1, ncols=n_cluster, figsize=(10, 4))
            cluster_colors = sns.color_palette('bright', n_cluster)  # Generate a color palette
            
                # dengan setiap subplot mewakili cluster yang berbeda.
            for i, cluster_label in enumerate(range(n_cluster)):
                #Mengambil subset data dari pca_data untuk cluster tertentu.
                cluster_df = pca_data[pca_data['Cluster'] == cluster_label]
                #kemudian Mengatur label sumbu x, y, dan judul untuk setiap subplot.
                sns.scatterplot(x=x_axis, y=y_axis, data=cluster_df, ax=axes[i], color=cluster_colors[i])
                axes[i].set_xlabel(x_axis)
                axes[i].set_ylabel(y_axis)
                axes[i].set_title(f"Cluster {cluster_label}")
                
                #Menampilkan subplot menggunakan st.pyplot() pada aplikasi Streamlit.
            st.pyplot(fig)
            #Jika checkbox "Show Subplots per Cluster" tidak dicentang atau jumlah cluster hanya 1,
        else:
            # maka akan ditampilkan pesan "Tidak ada perbedaan dengan Plot awal dan subplot".
            st.write("Tidak ada perbedaan dengan Plot awal dan subplot")


        #Membuat checkbox pada aplikasi Streamlit dengan label "Show The Table" untuk memilih apakah ingin menampilkan tabel atau tidak.
    if st.checkbox("Show The Table"):
        with st.container():
            st.write("Ini adalah Dataframe awal")
            df_bersih["Cluster"] = labels #enambahkan kolom "Cluster" pada dataframe df_bersih dengan nilai dari labels.
            st.write(df_bersih)#Menampilkan dataframe df_bersih pada aplikasi Streamlit.
            
            

from scipy.stats import chi2_contingency
import scipy.stats as stats
                
with tab2:
    with st.container():
        st.title("Elbow Method setiap cluster")
        inertias = []#Membuat sebuah list kosong untuk menyimpan nilai inertia dari setiap percobaan cluster.
        for k in range(1, 11):  # Melakukan iterasi untuk nilai k dari 1 hingga 10.
            kmeans = KMeans(n_clusters=k, random_state=42)#Membuat objek KMeans dengan jumlah cluster sebanyak k.
            kmeans.fit(pca_data)#Melakukan proses clustering menggunakan KMeans pada data PCA (pca_data).
            inertias.append(kmeans.inertia_) #Menyimpan nilai inertia dari clustering yang dilakukan ke dalam list inertias.

        # Membuat plot Elbow Method menggunakan sns.lineplot() dengan sumbu x adalah jumlah cluster dan sumbu y adalah nilai inertia.
        # Plot ini menunjukkan perubahan inertia terhadap jumlah cluster yang digunakan.
        fig, ax = plt.subplots()
        sns.lineplot(x=range(1, 11), y=inertias, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        # Display the plot using Streamlit
        st.pyplot(fig)

    #Membuat sebuah list yang berisi nama-nama kolom kategorikal yang akan dianalisis.
    categorical_columns = ['PT', 'STATUS KARYAWAN', 'JABATAN', 'LEVEL', 'DEPARTEMENT', 'BISNIS', 'LOKASI', 'JENIS TURNOVER', 'PLACEMENT', 'PAKLARING']
    correlations = {}#Membuat sebuah dictionary kosong untuk menyimpan hasil korelasi.

    with st.container():
        st.title("Korelasi Alasan resign dengan kategorikal kolum")
        # Melakukan iterasi melalui setiap kolom kategorikal.
        for column in categorical_columns:
            # Membuat tabel kontingensi antara kolom "ALASAN RESIGN" dan kolom kategorikal saat ini.
            cross_tab = pd.crosstab(df_bersih['ALASAN RESIGN'], df_bersih[column])

            # Menghitung chi-square test statistic, p-value, degrees of freedom, dan expected values berdasarkan tabel kontingensi.
            chi2, p, dof, expected = chi2_contingency(cross_tab)

            # Menghitung jumlah total pengamatan.
            n = cross_tab.sum().sum()  # Jumlah total pengamatan
            
            #Menghitung Cramér's V berdasarkan chi-square test statistic.
            cramers_v = np.sqrt(chi2 / (n * (min(cross_tab.shape) - 1)))

            # Menyimpan hasil korelasi ke dalam dictionary
            correlations[column] = {'Chi-square test statistic': chi2,
                                    'P-value': p,
                                    'Degrees of freedom': dof,
                                    "Cramér's V": cramers_v}
        
        # Membuat DataFrame correlation_df dari dictionary correlations.
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index')

        # Mengurutkan correlation_df berdasarkan nilai Cramér's V dalam urutan menurun
        correlation_df = correlation_df.sort_values("Cramér's V", ascending=False)

        # Membuat plot bar menggunakan sns.barplot() dengan sumbu x adalah Cramér's V dan sumbu y adalah kolom kategorikal. 
        # Plot ini menunjukkan korelasi antara kolom "ALASAN RESIGN" dengan setiap kolom kategorikal.
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Cramér's V", y=correlation_df.index, data=correlation_df, palette='viridis')
        plt.xlabel("Cramér's V")
        plt.ylabel('Categorical Columns')
        plt.title('Correlation between ALASAN RESIGN and Categorical Columns')

        # Menampilkan plot korelasi pada aplikasi Streamlit.
        st.pyplot(fig)  