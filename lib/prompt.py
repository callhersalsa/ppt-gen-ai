# Breaking down user's prompt into several queries
prompt_query = """
Break down the following sentence into separate search queries that capture its main points. Return each query as a separate line, and be concise.

Example
Input: Tell me the latest AMD stock price and how it compares to NVIDIA over the last month. Also, what are the major AI announcements from AMD recently?

Output:
- latest AMD stock price
- AMD vs NVIDIA stock performance last month
- recent AMD AI announcements

Sentence
Input: {prompt}
"""

# Querying whole content for PDF
prompt__pdf_content = """
Anda adalah seorang jurnalis. Tugas Anda adalah menulis artikel berdasarkan teks yang diberikan.

Aturan:
- Gunakan HANYA informasi dari teks di bawah ini sebagai sumber.
- Jangan menambah atau mengarang informasi di luar teks.
- Sertakan detail penting (nama, tanggal, angka, lokasi, kutipan) bila ada.
- Tulis artikel dalam format markdown.

Teks sumber:
{context}

Instruksi artikel:
Fokus bahasan: {query}
Topik: {topic}

Struktur artikel:
1. Judul utama yang menarik.
2. Pembukaan ringan dan relevan.
3. Pembahasan detail dan faktual (gunakan detail spesifik dari teks).
4. Kesimpulan atau pemikiran akhir.

Tuliskan dalam bahasa Indonesia yang jelas dan rapi.
"""

# Querying whole content for PPT
prompt__ppt_content = """
Anda adalah seorang pembuat konten presentasi. Tugas Anda adalah membuat ringkasan untuk slide PowerPoint (PPT) berdasarkan teks yang diberikan.

Aturan:
- Gunakan HANYA informasi dari teks di bawah ini.
- Jangan menambah atau mengarang informasi.
- Sertakan detail penting (nama, tanggal, angka, lokasi, kutipan) bila relevan.
- Output berupa teks biasa tanpa simbol tambahan seperti *, bullet, atau formatting aneh.
- Hasilkan hanya teks isi slide — TANPA kalimat pembuka, penjelasan, atau ucapan tambahan.

Teks sumber:
{context}

Instruksi PPT:
Topik umum: {query}
Fokus bahasan: {topic}

Format keluaran harus seperti berikut:

# [Judul yang menarik dan relevan dengan topik]

## [Judul Pengantar]
[Hal-hal menarik atau konteks singkat terkait topik]

## [Judul Slide 1]
[Isi dan poin-poin ringkas dan padat (detail penting, subtopik, data)]

## [Judul Slide 2]
[Isi dan poin-poin ringkas dan padat (detail penting, subtopik, data)]

## [Judul Slide 3]
[Isi dan poin-poin ringkas dan padat (detail penting, subtopik, data)]

## [Judul Slide 4]
[Isi dan poin-poin ringkas dan padat (detail penting, subtopik, data)]

## [Penutup]
[kesimpulan atau refleksi akhir]

Tiap slide hanya boleh berisi poin-poin atau penjelasan yang ringkas dan padat. Slide bisa lebih dari 4 jika diperlukan, tapi usahakan tetap fokus dan relevan.
"""