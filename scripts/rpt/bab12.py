"""Bab I Pendahuluan dan Bab II Landasan Teori."""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from . import engine
from .engine import (
    CTX, bullet, chapter, code_block, equation, figure, h2, h3, hr,
    numbered, p, table_block,
)

M = CTX.metrics
D = M["dataset"]


def bab1(ctx) -> None:
    ctx.add(*chapter("BAB I", "Pendahuluan"))

    ctx.add(h2("1.1  Latar Belakang"))
    ctx.add(p(
        "Industri telekomunikasi merupakan salah satu sektor yang paling cepat mengalami "
        "kejenuhan pasar. Jumlah pengguna telepon seluler di Indonesia telah melampaui jumlah "
        "penduduk, sehingga pertumbuhan pelanggan baru tidak lagi menjadi sumber pendapatan "
        "utama. Dalam kondisi seperti ini, mempertahankan pelanggan yang sudah ada menjadi "
        "prioritas strategis: setiap pelanggan yang bertahan memberikan arus pendapatan "
        "berulang, sementara setiap pelanggan yang pergi harus digantikan melalui proses "
        "akuisisi yang menelan biaya pemasaran, subsidi perangkat, dan diskon."
    ))
    ctx.add(p(
        "Pelanggan yang berhenti berlangganan atau berpindah ke penyedia lain dikenal dengan "
        "istilah <i>customer churn</i>. Persoalan utama churn bukan sekadar kehilangan satu "
        "transaksi, melainkan hilangnya nilai pelanggan di masa depan. Reichheld dan Sasser "
        "(1990) menunjukkan bahwa kenaikan kecil pada tingkat retensi dapat meningkatkan laba "
        "secara tidak proporsional, karena biaya melayani pelanggan lama jauh lebih rendah "
        "dibandingkan biaya menarik pelanggan baru. Gupta dan Lehmann (2006) menegaskan hal "
        "serupa dari perspektif <i>customer lifetime value</i>: nilai seorang pelanggan "
        "ditentukan oleh lamanya ia bertahan, bukan oleh besar transaksi pertamanya."
    ))
    ctx.add(p(
        "Yang menyulitkan, churn sering bersifat diam. Pelanggan tidak selalu menyampaikan "
        "keluhan sebelum berhenti; mereka hanya tidak memperpanjang kontrak atau berhenti "
        "mengisi ulang pulsa. Tanpa mekanisme peringatan dini, perusahaan baru menyadari "
        "kehilangan setelah churn terjadi, ketika peluang intervensi sudah tertutup. Karena "
        "itu, diperlukan model prediktif yang mampu menaksir probabilitas churn setiap "
        "pelanggan sebelum keputusan berhenti diambil, sehingga tim retensi dapat bertindak "
        "secara terarah dan terukur."
    ))
    ctx.add(p(
        "Berbagai algoritma pembelajaran mesin dapat digunakan untuk tugas tersebut, mulai dari "
        "regresi logistik, pohon keputusan, <i>random forest</i>, hingga <i>gradient boosting</i>. "
        "Meskipun model ansambel sering menghasilkan akurasi lebih tinggi, regresi logistik "
        "tetap menjadi pilihan utama dalam praktik bisnis karena tiga alasan. Pertama, keluarannya "
        "berupa probabilitas yang dapat langsung diterjemahkan menjadi skor risiko. Kedua, "
        "hubungan antar fitur dan target dapat dibaca melalui koefisien dan <i>odds ratio</i>, "
        "sehingga hasilnya dapat dipertanggungjawabkan kepada pihak non-teknis. Ketiga, model ini "
        "ringan, cepat dilatih ulang, dan menjadi <i>baseline</i> yang wajib dikalahkan oleh model "
        "yang lebih kompleks sebelum model tersebut diadopsi."
    ))
    ctx.add(p(
        "Berdasarkan uraian tersebut, kelompok kami memilih topik <b>Klasifikasi Logistic "
        "Regression</b> dengan studi kasus prediksi churn pada perusahaan telekomunikasi. Kami "
        "tidak berhenti pada pemodelan, tetapi membangun sistem yang utuh: pembersihan data, "
        "analisis eksploratif, pelatihan model, evaluasi dengan beberapa metrik, penentuan ambang "
        "keputusan yang selaras dengan kepentingan bisnis, serta penyajian daftar pelanggan "
        "berisiko tinggi melalui <i>dashboard</i>. Dengan demikian, laporan ini diharapkan tidak "
        "hanya menjelaskan cara kerja regresi logistik, tetapi juga menunjukkan bagaimana "
        "algoritma tersebut dipakai untuk menjawab persoalan nyata."
    ))

    ctx.add(h2("1.2  Rumusan Masalah"))
    ctx.add(p("Berdasarkan latar belakang di atas, rumusan masalah pada laporan ini adalah:"))
    ctx.add(*numbered([
        "Faktor-faktor apa yang paling berpengaruh terhadap keputusan pelanggan telekomunikasi "
        "untuk berhenti berlangganan?",
        "Seberapa baik algoritma <i>logistic regression</i> dalam memprediksi churn pelanggan, "
        "ditinjau dari metrik <i>accuracy</i>, <i>precision</i>, <i>recall</i>, <i>F1-score</i>, "
        "dan <i>ROC-AUC</i>?",
        "Ambang probabilitas (<i>threshold</i>) manakah yang paling tepat digunakan apabila model "
        "dioperasikan oleh tim retensi dengan kapasitas terbatas?",
        "Bagaimana hasil model diterjemahkan menjadi rancangan sistem dan rekomendasi strategi "
        "retensi yang dapat dijalankan?",
    ]))

    ctx.add(h2("1.3  Tujuan"))
    ctx.add(*numbered([
        "Mengidentifikasi fitur yang paling kuat memengaruhi churn beserta arah dan besar "
        "pengaruhnya melalui nilai koefisien dan <i>odds ratio</i>.",
        "Membangun dan mengevaluasi model klasifikasi churn berbasis regresi logistik pada data "
        "Telco Customer Churn.",
        "Menganalisis pergeseran <i>precision</i> dan <i>recall</i> pada berbagai ambang keputusan "
        "untuk menentukan ambang yang paling sesuai dengan tujuan retensi.",
        "Membangun sistem prediksi yang dapat dijalankan ulang, lengkap dengan <i>dashboard</i> "
        "pemantauan risiko pelanggan.",
    ]))

    ctx.add(h2("1.4  Manfaat"))
    ctx.add(h3("1.4.1  Manfaat Akademis"))
    ctx.add(p(
        "Laporan ini menyajikan penerapan regresi logistik secara utuh, mulai dari penurunan "
        "persamaan, interpretasi koefisien, hingga evaluasi pada data yang tidak seimbang. "
        "Dengan menyertakan kode program yang dapat dijalankan ulang, hasil penelitian ini dapat "
        "dijadikan acuan bagi mahasiswa yang ingin mempelajari alur kerja klasifikasi secara "
        "praktis."
    ))
    ctx.add(h3("1.4.2  Manfaat Praktis"))
    ctx.add(*[
        bullet("Bagi perusahaan telekomunikasi, model dapat dipakai untuk menyusun daftar prioritas "
               "pelanggan yang perlu dihubungi tim retensi setiap bulan."),
        bullet("Bagi tim pemasaran, hasil interpretasi fitur memberi petunjuk intervensi mana yang "
               "layak diuji, misalnya insentif migrasi kontrak atau perbaikan kualitas layanan."),
        bullet("Bagi pembaca umum, laporan ini menunjukkan bahwa model prediktif yang sederhana "
               "dapat memberi nilai bisnis apabila ambang keputusan dan metrik evaluasi dipilih "
               "secara tepat."),
    ])

    ctx.add(h2("1.5  Batasan Masalah"))
    ctx.add(*[
        bullet("Data yang digunakan bersifat <i>cross-sectional</i> (satu potret waktu), bukan "
               "deret waktu, sehingga model tidak menangkap perubahan perilaku pelanggan."),
        bullet("Dataset berasal dari satu perusahaan telekomunikasi Amerika Serikat dan bersifat "
               "anonim; karakteristik pelanggan Indonesia dapat berbeda."),
        bullet("Penelitian dibatasi pada satu algoritma, yaitu regresi logistik dengan regularisasi "
               "L2, tanpa pencarian hiperparameter yang menyeluruh."),
        bullet("Evaluasi difokuskan pada metrik klasifikasi, belum menghitung nilai rupiah dari "
               "keuntungan retensi (<i>expected maximum profit</i>)."),
    ])

    ctx.add(h2("1.6  Sistematika Penulisan"))
    ctx.add(*table_block(
        ["Bab", "Isi"],
        [
            ["BAB I  Pendahuluan",
             "Latar belakang, rumusan masalah, tujuan, manfaat, batasan, dan sistematika penulisan."],
            ["BAB II  Landasan Teori",
             "Data mining, regresi logistik, preprocessing, metrik evaluasi, dan penelitian terdahulu."],
            ["BAB III  Metodologi",
             "Deskripsi data, rancangan sistem, tahapan penelitian, dan lingkungan implementasi."],
            ["BAB IV  Hasil dan Pembahasan",
             "Analisis eksploratif, performa model, tuning ambang, interpretasi fitur, dan rekomendasi."],
            ["BAB V  Kesimpulan dan Saran",
             "Jawaban atas rumusan masalah, keterbatasan, dan arah pengembangan."],
        ],
        widths=[4.2 * cm, 11.5 * cm],
        caption="Sistematika penulisan laporan",
        label="Tabel 1.1",
    ))


def bab2(ctx) -> None:
    ctx.add(*chapter("BAB II", "Landasan Teori"))

    ctx.add(h2("2.1  Data Mining dan Klasifikasi"))
    ctx.add(p(
        "Data mining adalah proses menemukan pola yang bermakna dari kumpulan data berukuran "
        "besar melalui kombinasi statistik, pembelajaran mesin, dan basis data. Salah satu tugas "
        "utamanya adalah klasifikasi, yaitu membangun fungsi yang memetakan vektor fitur "
        "ke salah satu kelas diskret. Pada penelitian ini kelas yang dimaksud bersifat biner: "
        "pelanggan diprediksi <i>churn</i> (1) atau tidak <i>churn</i> (0)."
    ))
    ctx.add(p(
        "Secara formal, diberikan himpunan data latih berisi n pasangan (x<sub>i</sub>, y<sub>i</sub>) "
        "dengan x<sub>i</sub> ∈ R<super>k</super> dan y<sub>i</sub> ∈ {0, 1}. Tujuan "
        "pembelajaran adalah mencari fungsi f(x) yang mampu menaksir peluang P(y = 1 | x) seakurat "
        "mungkin, lalu memetakan peluang tersebut menjadi keputusan kelas melalui sebuah ambang."
    ))
    ctx.add(*table_block(
        ["Notasi", "Keterangan"],
        [
            ["n", "Jumlah observasi pada data latih"],
            ["k", "Jumlah fitur setelah transformasi"],
            ["x<sub>i</sub>", "Vektor fitur observasi ke-i"],
            ["y<sub>i</sub>", "Label kelas observasi ke-i (0 atau 1)"],
            ["β<sub>0</sub>", "Intersep (<i>intercept</i>) model"],
            ["β<sub>j</sub>", "Koefisien fitur ke-j"],
            ["p<sub>i</sub>", "Probabilitas prediksi bahwa observasi ke-i termasuk kelas 1"],
            ["τ", "Ambang keputusan (<i>threshold</i>)"],
            ["C", "Parameter regularisasi (kebalikan kekuatan penalti)"],
            ["η", "Laju pembelajaran (<i>learning rate</i>)"],
        ],
        widths=[2.6 * cm, 12.9 * cm],
        caption="Notasi matematis yang digunakan pada laporan",
        label="Tabel 2.1",
        align=["CENTER", "LEFT"],
    ))

    ctx.add(h2("2.2  Regresi Logistik"))
    ctx.add(h3("2.2.1  Mengapa Bukan Regresi Linear"))
    ctx.add(p(
        "Regresi linear menaksir nilai kontinu melalui kombinasi linear fitur:"))
    ctx.add(*equation(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k", "2.1"))
    ctx.add(p(
        "Apabila persamaan tersebut dipakai langsung untuk memprediksi probabilitas, muncul dua "
        "masalah mendasar. Pertama, nilai z tidak terbatas, sedangkan probabilitas wajib berada "
        "pada rentang 0 sampai 1. Kedua, hubungan antara fitur dan probabilitas dalam banyak kasus "
        "bersifat tidak linear: pengaruh satu satuan tambahan pada fitur akan mengecil ketika "
        "probabilitas sudah mendekati 0 atau 1. Regresi logistik mengatasi keduanya dengan "
        "mengapit nilai z menggunakan fungsi logistik."
    ))

    ctx.add(h3("2.2.2  Fungsi Sigmoid"))
    ctx.add(p(
        "Fungsi logistik (sigmoid) memetakan seluruh bilangan riil ke rentang (0, 1) secara monoton:"))
    ctx.add(*equation(r"p(\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}", "2.2"))
    ctx.add(p(
        "Sifat penting fungsi ini adalah bentuknya menyerupai huruf S: perubahan z pada daerah "
        "tengah memberi pengaruh besar terhadap p, sedangkan pada ekstrem pengaruhnya mengecil. "
        "Sifat tersebut sesuai dengan intuisi bisnis, bahwa penambahan satu bulan masa berlangganan "
        "lebih berarti bagi pelanggan baru daripada bagi pelanggan yang sudah bertahan lima tahun. "
        "Keputusan kelas kemudian diambil dengan membandingkan p terhadap ambang τ: pelanggan "
        "diprediksi churn apabila p ≥ τ."
    ))

    ctx.add(h3("2.2.3  Bentuk Log-Odds (Logit)"))
    ctx.add(p(
        "Kebalikan fungsi sigmoid adalah fungsi logit. Dengan menyusun ulang Persamaan (2.2), "
        "diperoleh hubungan linear antara fitur dan logaritma natural dari rasio peluang:"))
    ctx.add(*equation(
        r"\ln \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k", "2.3"))
    ctx.add(p(
        "Besaran p/(1−p) disebut <i>odds</i>, yaitu perbandingan peluang kejadian terjadi "
        "terhadap peluang tidak terjadi. Sebagai ilustrasi, pelanggan dengan probabilitas churn 0,80 "
        "memiliki odds 4 banding 1. Karena transformasi logit bersifat linear terhadap fitur, "
        "regresi logistik tetap tergolong model linear, namun linearitasnya berlaku pada skala "
        "log-odds, bukan pada skala probabilitas."
    ))

    ctx.add(h3("2.2.4  Estimasi Parameter"))
    ctx.add(p(
        "Parameter β diestimasi dengan <i>maximum likelihood estimation</i> (MLE), yaitu "
        "memilih nilai parameter yang paling besar kemungkinannya menghasilkan data yang diamati. "
        "Dengan asumsi antar observasi saling bebas, fungsi likelihood dirumuskan sebagai:"))
    ctx.add(*equation(
        r"L(\beta) = \prod_{i=1}^{n} p_i^{y_i}\,(1-p_i)^{1-y_i}", "2.4"))
    ctx.add(p(
        "Untuk memudahkan perhitungan, dilakukan transformasi logaritma sehingga perkalian berubah "
        "menjadi penjumlahan:"))
    ctx.add(*equation(
        r"\ell(\beta) = \sum_{i=1}^{n}\left[y_i \ln p_i + (1-y_i)\ln(1-p_i)\right]", "2.5"))
    ctx.add(p(
        "Memaksimalkan Persamaan (2.5) ekuivalen dengan meminimalkan nilai negatifnya, yang dikenal "
        "sebagai fungsi kerugian <i>cross-entropy</i> atau <i>log loss</i>:"))
    ctx.add(*equation(
        r"J(\beta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln p_i + (1-y_i)\ln(1-p_i)\right]", "2.6"))
    ctx.add(p(
        "Berbeda dengan regresi linear kuadrat terkecil, Persamaan (2.6) tidak memiliki solusi "
        "bentuk tertutup. Gradien fungsi kerugian terhadap parameter adalah:"))
    ctx.add(*equation(
        r"\nabla J(\beta) = \frac{1}{n}\sum_{i=1}^{n}\left(p_i - y_i\right)\mathbf{x}_i", "2.7"))
    ctx.add(p(
        "Karena itu penyelesaian dilakukan secara numerik, misalnya dengan <i>gradient descent</i> "
        "yang memperbarui parameter secara iteratif:"))
    ctx.add(*equation(
        r"\beta^{(t+1)} = \beta^{(t)} - \eta \,\nabla J\!\left(\beta^{(t)}\right)", "2.8"))
    ctx.add(p(
        "Implementasi pada penelitian ini menggunakan pustaka scikit-learn dengan pemecah "
        "<i>lbfgs</i>, yaitu varian metode kuasi-Newton yang memanfaatkan informasi turunan kedua "
        "secara aproksimasi sehingga konvergen lebih cepat dibandingkan <i>gradient descent</i> "
        "biasa. Model konvergen setelah 49 iterasi, jauh di bawah batas 1.000 iterasi yang "
        "ditetapkan."
    ))

    ctx.add(h3("2.2.5  Interpretasi melalui Odds Ratio"))
    ctx.add(p(
        "Karena model linear pada skala log-odds, eksponensial koefisien menghasilkan <i>odds "
        "ratio</i> (OR), yaitu faktor pengali odds ketika fitur bertambah satu satuan dengan "
        "asumsi fitur lain konstan:"))
    ctx.add(*equation(
        r"\mathrm{OR}_j = \frac{\mathrm{odds}(x_j + 1)}{\mathrm{odds}(x_j)} = e^{\beta_j}", "2.9"))
    ctx.add(p(
        "Nilai OR lebih besar dari 1 menandakan fitur menaikkan risiko churn, sedangkan nilai di "
        "bawah 1 menandakan fitur bersifat protektif. Contoh pembacaan: OR sebesar 0,265 berarti "
        "odds churn turun sekitar 73,5%, sedangkan OR sebesar 3,26 berarti odds churn naik 3,26 "
        "kali. Untuk fitur numerik yang telah distandardisasi, OR dibaca sebagai perubahan odds "
        "per kenaikan satu simpangan baku, bukan per satu satuan asli. Kemampuan interpretasi "
        "inilah yang membuat regresi logistik banyak dipakai pada domain yang menuntut "
        "keterjelasan, seperti kredit, asuransi, dan kesehatan."
    ))

    ctx.add(h3("2.2.6  Regularisasi L2"))
    ctx.add(p(
        "Apabila fitur saling berkorelasi, estimasi koefisien menjadi tidak stabil: nilai koefisien "
        "dapat membesar dengan tanda berlawanan. Regularisasi L2 menambahkan penalti kuadrat pada "
        "fungsi kerugian sehingga koefisien terdorong mendekati nol:"))
    ctx.add(*equation(
        r"J_{\mathrm{reg}}(\beta) = J(\beta) + \frac{1}{2C}\sum_{j=1}^{k}\beta_j^2", "2.10"))
    ctx.add(p(
        "Parameter C mengatur kekuatan penalti: nilai C kecil berarti penalti kuat dan model lebih "
        "konservatif. Penelitian ini menggunakan nilai bawaan C = 1,0. Regularisasi tidak "
        "menghilangkan korelasi antar fitur, sehingga interpretasi koefisien pada fitur yang "
        "berkorelasi tinggi tetap perlu hati-hati, sebagaimana dibahas pada Bab IV."
    ))

    ctx.add(h2("2.3  Praktik Terbaik Praprocessing Data"))
    ctx.add(h3("2.3.1  One-Hot Encoding"))
    ctx.add(p(
        "Regresi logistik hanya menerima masukan numerik, sedangkan sebagian besar fitur pada "
        "dataset ini bersifat kategorikal, misalnya tipe kontrak dan metode pembayaran. "
        "<i>One-hot encoding</i> mengubah satu fitur berkategori m menjadi m kolom biner. Apabila "
        "seluruh kolom dipertahankan, muncul kolinearitas sempurna yang disebut <i>dummy variable "
        "trap</i>, karena jumlah seluruh kolom selalu sama dengan satu. Penelitian ini membuang "
        "satu kategori acuan (<i>drop first</i>) sehingga setiap kategori dibandingkan terhadap "
        "kategori acuan tersebut. Sebagai konsekuensinya, koefisien kategori acuan tidak muncul "
        "secara eksplisit dan menjadi pembanding bagi kategori lain."
    ))
    ctx.add(h3("2.3.2  Standardisasi"))
    ctx.add(p(
        "Fitur numerik pada dataset ini memiliki skala yang sangat berbeda: masa berlangganan "
        "berkisar 0 sampai 72 bulan, sedangkan total tagihan mencapai ribuan dolar. Tanpa "
        "penyeragaman skala, koefisien tidak dapat dibandingkan dan optimasi menjadi kurang stabil. "
        "Standardisasi z-score mentransformasi setiap fitur menjadi:"))
    ctx.add(*equation(r"z = \frac{x - \mu}{\sigma}", "2.11"))
    ctx.add(p(
        "Setelah transformasi, rata-rata fitur menjadi 0 dan simpangan bakunya 1. Perlu ditegaskan "
        "bahwa standardisasi tidak mengubah bentuk distribusi maupun urutan nilai, hanya skala "
        "pembacaannya."
    ))
    ctx.add(h3("2.3.3  Pembagian Data dan Pencegahan Kebocoran Informasi"))
    ctx.add(p(
        "Data dibagi menjadi bagian latih dan bagian uji. Pembagian dilakukan secara "
        "<i>stratified</i>, yaitu proporsi kelas churn dijaga sama pada kedua bagian, agar "
        "evaluasi tidak bias ketika kelas tidak seimbang. Seluruh proses transformasi dibungkus "
        "dalam sebuah <i>pipeline</i> dan hanya di-<i>fit</i> pada data latih. Langkah ini penting "
        "untuk mencegah <i>data leakage</i>, yaitu kondisi ketika informasi dari data uji ikut "
        "mempengaruhi proses pelatihan sehingga hasil evaluasi terlihat lebih baik daripada "
        "kenyataan."
    ))

    ctx.add(h2("2.4  Evaluasi Model Klasifikasi"))
    ctx.add(h3("2.4.1  Confusion Matrix"))
    ctx.add(p(
        "Confusion matrix merangkum empat kemungkinan hasil prediksi: <i>true positive</i> (TP, "
        "churn diprediksi churn), <i>true negative</i> (TN, tidak churn diprediksi tidak churn), "
        "<i>false positive</i> (FP, tidak churn tetapi diprediksi churn), dan <i>false negative</i> "
        "(FN, churn tetapi diprediksi tidak churn). Seluruh metrik turunan pada bagian berikutnya "
        "dihitung dari keempat sel tersebut."
    ))
    ctx.add(h3("2.4.2  Accuracy, Precision, Recall, dan F1-Score"))
    ctx.add(p("Empat metrik utama didefinisikan sebagai berikut:"))
    ctx.add(*equation(r"\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}", "2.12"))
    ctx.add(*equation(r"\mathrm{Precision} = \frac{TP}{TP + FP}", "2.13"))
    ctx.add(*equation(r"\mathrm{Recall} = \frac{TP}{TP + FN}", "2.14"))
    ctx.add(*equation(
        r"F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}",
        "2.15"))
    ctx.add(p(
        "<i>Accuracy</i> mengukur proporsi prediksi yang benar secara keseluruhan, namun metrik ini "
        "menyesatkan pada data tidak seimbang. Apabila 73% pelanggan tidak churn, sebuah model yang "
        "selalu menjawab tidak churn sudah memperoleh <i>accuracy</i> 73% tanpa mempelajari apa pun. "
        "<i>Precision</i> menjawab pertanyaan seberapa dapat dipercaya peringatan yang dikeluarkan "
        "model, sedangkan <i>recall</i> menjawab seberapa besar pelanggan berisiko yang berhasil "
        "tertangkap. <i>F1-score</i> adalah rata-rata harmonik keduanya dan berguna ketika kedua "
        "jenis kesalahan dianggap sama penting."
    ))
    ctx.add(h3("2.4.3  ROC-AUC"))
    ctx.add(p(
        "Kurva ROC memetakan <i>true positive rate</i> terhadap <i>false positive rate</i> pada "
        "seluruh nilai ambang. Area di bawah kurva (AUC) memiliki makna probabilistik: peluang "
        "bahwa seorang pelanggan yang benar-benar churn memperoleh skor risiko lebih tinggi "
        "daripada pelanggan yang tidak churn, diambil secara acak. Nilai 0,5 berarti model tidak "
        "lebih baik daripada tebakan acak, sedangkan nilai mendekati 1 berarti diskriminasi "
        "hampir sempurna. Fawcett (2006) mengelompokkan AUC 0,8 sampai 0,9 sebagai diskriminasi "
        "yang baik. Keunggulan AUC adalah sifatnya yang bebas ambang, sehingga cocok untuk "
        "membandingkan model sebelum ambang operasional ditentukan."
    ))
    ctx.add(h3("2.4.4  Ketidakseimbangan Kelas dan Pemilihan Ambang"))
    ctx.add(p(
        "Pada data tidak seimbang, ambang bawaan 0,50 belum tentu optimal secara bisnis. "
        "Menurunkan ambang membuat model lebih mudah menyatakan churn, sehingga <i>recall</i> naik "
        "dan <i>precision</i> turun. Karena biaya <i>false negative</i> pada konteks retensi "
        "(kehilangan pelanggan secara permanen) umumnya lebih besar daripada biaya <i>false "
        "positive</i> (pemberian promo kepada pelanggan yang sebenarnya loyal), ambang yang lebih "
        "rendah sering kali lebih menguntungkan. Keputusan ini bersifat manajerial dan harus "
        "mempertimbangkan kapasitas tim retensi."
    ))

    ctx.add(h2("2.5  Kajian Penelitian Terdahulu"))
    ctx.add(p(
        "Verbeke, Martens, Mues, dan Baesens (2012) meneliti prediksi churn pada sektor "
        "telekomunikasi dengan pendekatan berbasis profit. Mereka menunjukkan bahwa pemilihan "
        "teknik klasifikasi sebaiknya tidak semata-mata didasarkan pada akurasi, tetapi pada "
        "dampak ekonominya, dan memperkenalkan ukuran <i>expected maximum profit</i> untuk "
        "mengevaluasi model pada ambang yang berbeda. Temuan tersebut menjadi salah satu alasan "
        "mengapa laporan ini tidak berhenti pada satu nilai ambang, melainkan menyajikan tabel "
        "pergeseran <i>precision</i> dan <i>recall</i> pada berbagai ambang."
    ))
    ctx.add(p(
        "Ascarza (2018) menguji efektivitas program retensi melalui eksperimen lapangan dan "
        "menemukan bahwa menyasar pelanggan dengan risiko churn tertinggi tidak selalu efektif, "
        "karena sebagian pelanggan berisiko tinggi justru tidak responsif terhadap insentif. "
        "Implikasinya, skor risiko sebaiknya dipadukan dengan analisis responsibilitas; model "
        "prediksi churn menjawab pertanyaan siapa yang berisiko, bukan siapa yang dapat "
        "dipertahankan."
    ))
    ctx.add(p(
        "Dari sisi metode, Hosmer, Lemeshow, dan Sturdivant (2013) merangkum praktik estimasi, "
        "uji kecocokan model, serta interpretasi <i>odds ratio</i> pada regresi logistik, termasuk "
        "peringatan mengenai ketidakstabilan koefisien ketika terjadi multikolinearitas. Chawla "
        "dkk. (2002) memperkenalkan SMOTE sebagai alternatif penanganan kelas tidak seimbang, "
        "sedangkan Fawcett (2006) memberikan rujukan baku untuk analisis ROC. Ketiga rujukan "
        "tersebut menjadi dasar pilihan metode pada Bab III serta pembahasan keterbatasan pada "
        "Bab IV dan Bab V."
    ))
    ctx.add(p(
        "Dibandingkan penelitian di atas, laporan ini tidak mengusulkan algoritma baru. "
        "Kontribusinya bersifat terapan: menyusun alur kerja regresi logistik yang dapat "
        "direproduksi, menafsirkan koefisien secara hati-hati dengan memperhatikan korelasi antar "
        "fitur, dan menerjemahkan hasilnya menjadi sistem skoring serta <i>dashboard</i> yang siap "
        "dipakai."
    ))
    ctx.add(Spacer(1, 0.3 * cm))
    ctx.add(hr(width="35%", space_before=8, space_after=4))