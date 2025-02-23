import os
import re
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ChatbotKesehatanMental:
    def __init__(self, json_path, context_path='models/data_konteks.json', max_words=6000, max_len=60, model_dir='models'):
        # Konfigurasi dasar
        self.json_path = json_path
        self.context_path = context_path
        self.max_words = max_words
        self.max_len = max_len
        self.model_dir = model_dir

        # Komponen model neural network
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None

        # Daftar stopwords bahasa Indonesia
        self.stopwords_indonesia = [
            'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
            'ini', 'itu', 'juga', 'sudah', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita',
            'akan', 'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'tentang', 'seperti',
            'ketika', 'bagi', 'sampai', 'karena', 'jika', 'namun', 'sehingga', 'yaitu',
            'yakni', 'menurut', 'hampir', 'dimana', 'bagaimana', 'selama', 'siapa',
            'mengapa', 'kapan', 'kemudian', 'sangat', 'hal', 'harus', 'sangat', 'sebuah',
            'tetap', 'setiap', 'banyak', 'sebagai', 'para', 'bahwa', 'lain', 'sering',
            'telah', 'lalu', 'tersebut', 'dahulu', 'belum', 'apakah'
        ]
        
        # Komponen TF-IDF untuk similarity
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words=self.stopwords_indonesia
        )
        self.tfidf_matrix = None
        self.patterns_vectorized = None

        # Data
        self.data_training = []
        self.data_umpan_balik = []
        self.concern_patterns = []
        self.concern_responses = []

        # Konteks percakapan
        self.konteks_percakapan = {
            'topik_saat_ini': None,
            'input_terakhir': None,
            'rantai_konteks': []
        }

        # Setup direktori dan path
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'model_chatbot.h5')
        self.tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        self.encoder_path = os.path.join(model_dir, 'label_encoder.json')
        self.vectorizer_path = os.path.join(model_dir, 'vectorizer.json')
        self.feedback_path = os.path.join(model_dir, 'data_umpan_balik.json')

        # Inisialisasi
        self.muat_data_training()
        self.muat_data_umpan_balik()
        self.muat_data_konteks()
        self.muat_atau_buat_model()
        self.inisialisasi_concern_similarity()

    def muat_data_konteks(self):
        """Memuat data konteks dari file"""
        try:
            if os.path.exists(self.context_path):
                with open(self.context_path, 'r', encoding='utf-8') as file:
                    data_konteks = json.load(file)
                    self.konteks_percakapan = data_konteks
            else:
                self.konteks_percakapan = {
                    'topik_saat_ini': None,
                    'input_terakhir': None,
                    'rantai_konteks': []
                }
        except Exception as e:
            print(f"Kesalahan memuat konteks: {e}")
            self.konteks_percakapan = {
                'topik_saat_ini': None,
                'input_terakhir': None,
                'rantai_konteks': []
            }

    def simpan_data_konteks(self):
        """Menyimpan data konteks ke file"""
        try:
            with open(self.context_path, 'w', encoding='utf-8') as file:
                json.dump(self.konteks_percakapan, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Kesalahan menyimpan konteks: {e}")

    def praproses_teks(self, teks):
        """Pra-proses teks input"""
        teks = str(teks).lower()
        # Hapus tanda tanya di akhir kalimat tapi simpan di tengah
        if teks.endswith('?'):
            teks = teks[:-1]
        # Hapus karakter khusus kecuali tanda tanya di tengah kata/kalimat
        teks = re.sub(r'[^\w\s\?]', '', teks)
        # Hapus spasi berlebih
        teks = ' '.join(teks.split())
        return teks

    def muat_data_training(self):
        """Memuat data training dari file JSON"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                self.data_training = json_data.get('data', [])

                for intent in self.data_training:
                    intent['patterns'] = [
                        self.praproses_teks(pattern)
                        for pattern in intent['patterns']
                    ]
        except Exception as e:
            print(f"Kesalahan memuat data training: {e}")
            self.data_training = []

    def muat_data_umpan_balik(self):
        """Memuat data umpan balik dari file"""
        try:
            if os.path.exists(self.feedback_path):
                with open(self.feedback_path, 'r', encoding='utf-8') as file:
                    self.data_umpan_balik = json.load(file)
            else:
                self.data_umpan_balik = []
        except Exception as e:
            print(f"Kesalahan memuat umpan balik: {e}")
            self.data_umpan_balik = []

    def simpan_data_umpan_balik(self):
        """Menyimpan data umpan balik ke file"""
        try:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(self.feedback_path, 'w', encoding='utf-8') as file:
                json.dump(self.data_umpan_balik, file, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Kesalahan menyimpan umpan balik: {e}")

    def tambah_umpan_balik(self, input_pengguna, respons_pengguna, respons_bot):
        """Menambahkan umpan balik baru"""
        input_diproses = self.praproses_teks(input_pengguna)
        respons_diproses = self.praproses_teks(respons_pengguna)

        # Cek apakah input sudah ada
        for feedback in self.data_umpan_balik:
            if self.praproses_teks(feedback['input']) == input_diproses:
                feedback['respons_yang_diharapkan'] = respons_diproses
                self.simpan_data_umpan_balik()
                print("Chatbot: Terima kasih! Saya telah memperbarui respons untuk pertanyaan ini.")
                return

        # Tambah umpan balik baru
        entri_umpan_balik = {
            'input': input_diproses,
            'respons_yang_diharapkan': respons_diproses,
            'respons_bot': respons_bot,
            'timestamp': tf.timestamp().numpy(),
            'tag': 'umpan_balik'
        }

        self.data_umpan_balik.append(entri_umpan_balik)
        self.simpan_data_umpan_balik()
        print(f"Chatbot: Terima kasih! Saya telah mempelajari bahwa untuk pertanyaan '{input_pengguna}', saya harus menjawab '{respons_pengguna}'.")

    def bangun_model(self, ukuran_vocab, jumlah_kelas):
        """Membangun arsitektur model neural network"""
        model = Sequential([
            Embedding(ukuran_vocab, 256, input_length=self.max_len),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(jumlah_kelas, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def latih(self):
        """Melatih model chatbot"""
        try:
            # Persiapkan data
            patterns = []
            tags = []

            for intent in self.data_training:
                for pattern in intent['patterns']:
                    patterns.append(pattern)
                    tags.append(intent['tag'])

            # Tambahkan data umpan balik
            for feedback in self.data_umpan_balik:
                patterns.append(feedback['input'])
                tags.append(feedback['tag'])

            if not patterns:
                print("Tidak ada data training yang tersedia")
                return

            # Tokenisasi
            self.tokenizer.fit_on_texts(patterns)
            X = self.tokenizer.texts_to_sequences(patterns)
            X = pad_sequences(X, maxlen=self.max_len, padding='post')
            y = self.label_encoder.fit_transform(tags)

            # Bagi data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            # Bangun dan latih model
            ukuran_vocab = len(self.tokenizer.word_index) + 1
            jumlah_kelas = len(self.label_encoder.classes_)

            self.model = self.bangun_model(ukuran_vocab, jumlah_kelas)

            # Bobot kelas
            bobot_kelas = {}
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            for i, count in zip(unique, counts):
                bobot_kelas[i] = total / (len(unique) * count)

            # Latih
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=16,
                class_weight=bobot_kelas,
                callbacks=[
                    ModelCheckpoint(
                        self.model_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        save_weights_only=False
                    )
                ],
                verbose=1
            )

            # Simpan komponen
            self.model.save(self.model_path)
            
            # Simpan tokenizer dalam format JSON
            tokenizer_json = {
                'word_index': self.tokenizer.word_index,
                'word_counts': self.tokenizer.word_counts,
                'document_count': self.tokenizer.document_count,
                'word_docs': self.tokenizer.word_docs,
                'filters': self.tokenizer.filters,
                'split': self.tokenizer.split,
                'lower': self.tokenizer.lower,
                'num_words': self.tokenizer.num_words,
                'oov_token': self.tokenizer.oov_token
            }
            with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

            # Simpan label encoder dalam format JSON
            label_encoder_json = {
                'classes': self.label_encoder.classes_.tolist(),
                'class_mapping': {i: label for i, label in enumerate(self.label_encoder.classes_)}
            }
            with open(self.encoder_path, 'w', encoding='utf-8') as f:
                json.dump(label_encoder_json, f, ensure_ascii=False, indent=2)

            print("Model dan komponen berhasil dilatih dan disimpan")

        except Exception as e:
            print(f"Kesalahan saat training: {e}")
            import traceback
            traceback.print_exc()

    def muat_atau_buat_model(self):
        """Memuat model yang ada atau membuat model baru"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)

                # Muat tokenizer dari JSON
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_json = json.load(f)
                    
                self.tokenizer = Tokenizer(num_words=tokenizer_json['num_words'], 
                                         oov_token=tokenizer_json['oov_token'])
                self.tokenizer.word_index = tokenizer_json['word_index']
                self.tokenizer.word_counts = tokenizer_json['word_counts']
                self.tokenizer.document_count = tokenizer_json['document_count']
                self.tokenizer.word_docs = tokenizer_json['word_docs']
                self.tokenizer.filters = tokenizer_json['filters']
                self.tokenizer.split = tokenizer_json['split']
                self.tokenizer.lower = tokenizer_json['lower']

                # Muat label encoder dari JSON
                with open(self.encoder_path, 'r', encoding='utf-8') as f:
                    label_encoder_json = json.load(f)
                    
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(label_encoder_json['classes'])

                print("Model dan komponen berhasil dimuat")
            else:
                print("Model tidak ditemukan. Silakan latih model terlebih dahulu.")
        except Exception as e:
            print(f"Kesalahan memuat model: {e}")
            import traceback
            traceback.print_exc()

    def inisialisasi_concern_similarity(self):
        """Inisialisasi pola dan respons untuk kekhawatiran"""
        self.concern_patterns = [
            "saya merasa sedih",
            "saya merasa tertekan",
            "saya merasa cemas",
            "saya merasa stress",
            "saya khawatir",
            "saya takut",
            "saya bingung",
            "saya frustasi",
            "saya kesepian",
            "saya putus asa",
            "saya depresi",
            "saya ingin menyerah",
            "hidup terasa berat",
            "saya tidak sanggup lagi",
            "saya merasa tidak berguna",
            "saya merasa sendiri",
            "saya merasa gagal",
            "saya tidak bisa tidur",
            "saya kehilangan semangat",
            "saya merasa terbebani"
        ]
        
        self.concern_responses = [
            "Saya mengerti Anda sedang mengalami masa sulit. Maukah Anda bercerita lebih lanjut?",
            "Perasaan Anda valid. Mari kita bicarakan apa yang membuat Anda merasa seperti ini.",
            "Saya di sini untuk mendengarkan. Bagaimana pengalaman Anda menghadapi situasi ini?",
            "Terima kasih sudah mau berbagi. Bagaimana cara saya bisa membantu Anda saat ini?",
            "Saya memahami ini bukan hal yang mudah. Boleh ceritakan lebih detail?",
            "Anda tidak sendiri dalam menghadapi ini. Bagaimana perasaan Anda saat ini?",
            "Saya mendengarkan Anda dengan penuh perhatian. Apa yang membuat Anda merasa demikian?",
            "Penting untuk berbagi perasaan Anda. Bisakah Anda cerita lebih lanjut?",
            "Saya di sini untuk mendukung Anda. Apa yang bisa saya bantu saat ini?",
            "Mari kita hadapi ini bersama. Bagaimana situasi yang Anda alami?"
        ]

        # Vectorize concern patterns
        self.patterns_vectorized = self.vectorizer.fit_transform(self.concern_patterns)

    def hitung_similarity_kekhawatiran(self, input_text):
        """Menghitung similarity antara input dengan pola kekhawatiran"""
        input_vectorized = self.vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vectorized, self.patterns_vectorized)
        max_similarity = np.max(similarities)
        return max_similarity

    def dapatkan_respons_kekhawatiran(self, similarity_score):
        """Mendapatkan respons berdasarkan tingkat kekhawatiran"""
        if similarity_score > 0.6:  # Threshold tinggi untuk kekhawatiran yang jelas
            return random.choice(self.concern_responses)
        return None

    def prediksi_intensi(self, teks):
        """Memprediksi intensi dari input teks"""
        try:
            if not self.model or not hasattr(self.tokenizer, 'word_index'):
                return None

            # Cek di data umpan balik dulu
            teks_diproses = self.praproses_teks(teks)
            for feedback in self.data_umpan_balik:
                if self.praproses_teks(feedback['input']) == teks_diproses:
                    return {
                        'tag': 'umpan_balik',
                        'confidence': 1.0,
                        'response': feedback['respons_yang_diharapkan']
                    }

            # Jika tidak ada di umpan balik, gunakan model
            sequence = self.tokenizer.texts_to_sequences([teks_diproses])
            padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
            prediction = self.model.predict(padded)[0]
            predicted_class = prediction.argmax()
            confidence = prediction[predicted_class]

            if confidence >= 0.5:
                predicted_tag = self.label_encoder.inverse_transform([predicted_class])[0]
                for intent in self.data_training:
                    if intent['tag'] == predicted_tag:
                        return {
                            'tag': predicted_tag,
                            'confidence': float(confidence),
                            'response': random.choice(intent['responses'])
                        }

            return None

        except Exception as e:
            print(f"Kesalahan dalam prediksi: {e}")
            return None

    def get_response(self, input_pengguna):
        """Mendapatkan respons untuk input pengguna"""
        # Cek similarity dengan pola kekhawatiran
        similarity_score = self.hitung_similarity_kekhawatiran(input_pengguna)
        respons_kekhawatiran = self.dapatkan_respons_kekhawatiran(similarity_score)
        
        if respons_kekhawatiran:
            return {
                'tag': 'kekhawatiran',
                'confidence': float(similarity_score),
                'response': respons_kekhawatiran,
                'is_contextual': True
            }

        # Cek respons kontekstual
        respons_kontekstual = self.cari_respons_kontekstual(input_pengguna)
        if respons_kontekstual:
            self.perbarui_konteks(input_pengguna, respons_kontekstual['tag'])
            return respons_kontekstual

        # Prediksi dengan model
        prediksi = self.prediksi_intensi(input_pengguna)
        if prediksi:
            self.perbarui_konteks(input_pengguna, prediksi['tag'])
            return {
                'tag': prediksi['tag'],
                'confidence': prediksi['confidence'],
                'response': prediksi['response'],
                'matched_pattern': input_pengguna,
                'is_contextual': False
            }

        # Jika tidak ada kecocokan, berikan respons tidak diketahui
        pertanyaan_klarifikasi = [
            "Maaf, saya belum familiar dengan topik ini. Bisakah Anda menjelaskan lebih detail?",
            "Saya ingin belajar lebih banyak tentang hal ini. Bisa tolong jelaskan maksud Anda?",
            "Ini topik baru untuk saya. Bisakah Anda mengajarkan saya cara merespons yang tepat?",
            "Saya belum tahu jawaban yang tepat. Menurut Anda, bagaimana seharusnya saya merespons?"
        ]

        return {
            'tag': 'tidak_diketahui',
            'confidence': 0.0,
            'response': random.choice(pertanyaan_klarifikasi),
            'matched_pattern': input_pengguna,
            'is_contextual': False,
            'needs_learning': True
        }

    def perbarui_konteks(self, input_teks, intent_tag):
        """Memperbarui konteks percakapan"""
        self.konteks_percakapan['rantai_konteks'].append({
            'input': input_teks,
            'tag': intent_tag
        })

        if len(self.konteks_percakapan['rantai_konteks']) > 5:
            self.konteks_percakapan['rantai_konteks'].pop(0)

        self.konteks_percakapan['topik_saat_ini'] = intent_tag
        self.konteks_percakapan['input_terakhir'] = input_teks

        self.simpan_data_konteks()

    def cari_respons_kontekstual(self, input_saat_ini):
        """Mencari respons berdasarkan konteks percakapan"""
        if not self.konteks_percakapan['topik_saat_ini']:
            return None

        topik = self.konteks_percakapan['topik_saat_ini']
        input_terakhir = self.konteks_percakapan['input_terakhir']

        # Definisikan respons kontekstual
        topik_kontekstual = {
            'tidur': [
                {
                    'patterns': ['sulit tidur', 'tidak bisa tidur', 'insomnia'],
                    'follow_up': [
                        "Apakah Anda sudah mencoba menerapkan kebersihan tidur yang baik?",
                        "Baiklah, kebersihan tidur adalah praktik yang membantu meningkatkan kualitas tidur. Beberapa tips meliputi:",
                        "Pertama, atur jadwal tidur yang konsisten. Tidur dan bangun pada waktu yang sama setiap hari.",
                        "Hindari penggunaan elektronik sebelum tidur karena cahaya biru dapat mengganggu produksi melatonin.",
                        "Ciptakan lingkungan tidur yang nyaman: gelap, tenang, dan sejuk.",
                        "Hindari konsumsi kafein, alkohol, dan makanan berat menjelang tidur.",
                        "Lakukan relaksasi sebelum tidur seperti meditasi atau membaca buku."
                    ]
                },
                {
                    'patterns': ['belum'],
                    'follow_up': [
                        "Mari kita bahas beberapa strategi untuk meningkatkan kualitas tidur Anda.",
                        "Pertama, cobalah membuat rutinitas tidur yang teratur.",
                        "Gunakan teknik relaksasi seperti pernapasan dalam atau meditasi sebelum tidur.",
                        "Hindari penggunaan ponsel atau komputer minimal 1 jam sebelum tidur.",
                        "Jika masih sulit, pertimbangkan untuk berkonsultasi dengan profesional kesehatan."
                    ]
                }
            ],
            'kecemasan': [
                {
                    'patterns': ['cemas', 'khawatir', 'takut'],
                    'follow_up': [
                        "Saya mengerti Anda sedang merasa cemas. Sudahkah Anda mencoba teknik pernapasan?",
                        "Mari kita coba teknik pernapasan sederhana:",
                        "Tarik napas dalam-dalam selama 4 hitungan",
                        "Tahan napas selama 4 hitungan",
                        "Hembuskan napas perlahan selama 4 hitungan",
                        "Ulangi beberapa kali sampai Anda merasa lebih tenang"
                    ]
                }
            ],
            'stress': [
                {
                    'patterns': ['stress', 'tertekan', 'berat'],
                    'follow_up': [
                        "Bisakah Anda ceritakan lebih detail apa yang membuat Anda merasa stress?",
                        "Mari kita coba identifikasi pemicu stress Anda",
                        "Kemudian kita bisa mencari solusi yang sesuai bersama-sama"
                    ]
                }
            ]
        }

        teks_diproses = self.praproses_teks(input_saat_ini)

        if topik in topik_kontekstual:
            for item_konteks in topik_kontekstual[topik]:
                for pattern in item_konteks.get('patterns', []):
                    if pattern.lower() in input_terakhir.lower():
                        return {
                            'tag': topik,
                            'response': item_konteks['follow_up'][0],
                            'context_responses': item_konteks['follow_up'][1:],
                            'is_contextual': True
                        }

        return None

    def chat_interaktif(self):
        """Memulai sesi chat interaktif"""
        if not self.model:
            print("Chatbot: Model belum dilatih. Silakan latih model terlebih dahulu dengan mode 'latih'")
            return

        print("Chatbot: Hai! Saya siap membantu Anda. Silakan ceritakan apa yang Anda rasakan.")

        while True:
            input_pengguna = input("Anda: ")

            if input_pengguna.lower() in ['keluar', 'exit', 'bye', 'selesai']:
                print("Chatbot: Terima kasih telah berbicara dengan saya. Sampai jumpa!")
                break

            hasil = self.get_response(input_pengguna)

            if hasil['tag'] == 'selesai':
                print(f"Chatbot: {hasil['response']}")
                break

            print(f"Chatbot: {hasil['response']}")

            if hasil.get('is_contextual', False):
                context_responses = hasil.get('context_responses', [])
                for resp in context_responses:
                    lanjut = input("Lanjutkan? (ya/tidak): ").lower()
                    if lanjut == 'ya':
                        print(f"Chatbot: {resp}")
                    else:
                        break

            if not hasil.get('is_contextual', False):
                print(f"(Kategori: {hasil['tag']}, Keyakinan: {hasil['confidence']:.2f})")

            # Jika respons membutuhkan pembelajaran
            if hasil.get('needs_learning', False):
                user_response = input("Anda: ")
                if user_response.lower() not in ['keluar', 'exit', 'bye', 'tidak', 'no', 'selesai']:
                    self.tambah_umpan_balik(input_pengguna, user_response, hasil['response'])


def main():
    """Fungsi utama program"""
    mode = input("Pilih mode (latih/chat): ").lower()
    chatbot = ChatbotKesehatanMental('data.json')

    if mode == 'latih':
        chatbot.latih()
    elif mode == 'chat':
        chatbot.chat_interaktif()
    else:
        print("Mode tidak valid. Gunakan 'latih' atau 'chat'.")

if __name__ == "__main__":
    main()