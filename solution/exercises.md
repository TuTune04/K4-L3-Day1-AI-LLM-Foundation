# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi tăng temperature dần từ 0.0, mô hình đưa ra câu trả lời là các sự thật hiển nhiên (Hang Son Doong). Khi tăng lên 1.0 thì câu trả lời có văn phong cảm xúc hơn, đến 1.5 thì diễn tả lan man hơn *

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature trong khoảng từ 0.5 tới 1.0 vì để vừa đảm bảo tính trung thực của thông tin trả lời và câu trả lời không bị cứng ngắc khiến khách hàng khó chịu*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> * Tổng số tokens đâu ra mỗi ngày là 10500000 tokens, với GPT-4o : 105$ mỗi ngày, GPT-4o-mini : 6,3$ mỗi ngày, GPT-4o đặt gấp ~ 16,67 lần GPT-4o-mini; Nên dùng GPT-4o khi các tác vụ cần suy luận phức tạp, độ chính xác cao và không được phép sai sót; Nên dùng GPT-4o-mini khi các tác vụ lặp lại với khối lượng lớn, cấu trúc đơn giản (như phân loại ý định người dùng, tóm tắt văn bản ngắn, chatbot CSKH cơ bản) để tối ưu chi phí*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Khác biệt rõ nhất nằm ở góc tiếp cận và cách dùng từ: bản giáo viên viết rất gọn, chọn ví dụ gần gũi như cuốn sổ ghi điểm chung của lớp để đứa trẻ 8 tuổi cũng hình dung được; còn bản chuyên gia thì dài hơn hẳn, đi thẳng vào bản chất kỹ thuật với hàng loạt thuật ngữ tài chính. System prompt đóng vai trò như một khung định hướng, quyết định mô hình sẽ nhập vai ai và nói chuyện với ai, từ đó tự điều chỉnh từ vựng, độ sâu kiến thức và văn phong cho phù hợp.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với đoạn văn tiếng Việt 100 từ, công thức ước lượng lấy số từ chia cho 0,75 cho ra khoảng 133 token, trong khi bộ mã hóa tiktoken đếm thực tế lên tới khoảng 185 token, chênh lệch thực tế cao hơn ước lượng khoảng 39%. Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài do bảng từ vựng của tokenizer được tối ưu chủ yếu trên ngữ liệu tiếng Anh, các ký tự có dấu thanh tiếng Việt chiếm nhiều byte UTF-8 nên bị xé nhỏ thành nhiều mảnh token rời rạc, cộng thêm đặc điểm từ ngữ viết tách rời từng âm tiết khiến số lượng mảnh ghép mã hóa tăng lên đáng kể.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming tối quan trọng cho các ứng dụng tương tác trực tiếp như chatbot, trợ lý ảo hay sinh bài viết dài, giúp người dùng thấy ngay phản hồi qua từng token để giảm thời gian chờ đợi. Non-streaming phù hợp hơn với các tác vụ chạy ngầm, xử lý dữ liệu hàng loạt (batch jobs), chấm điểm tự động hoặc sinh cấu trúc JSON cần nhận trọn vẹn kết quả để parse và lưu trữ vào cơ sở dữ liệu.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff tự động kéo giãn thời gian chờ theo cấp số nhân qua mỗi lần thất bại, tạo ra khoảng nghỉ đủ dài để hệ thống máy chủ kịp giải phóng hàng đợi và phục hồi tài nguyên thay vì liên tục chịu tải dồn dập.  *

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Tôi chọn persona: "Bạn là trợ giảng lập trình AI thực tế, hãy trả lời bằng tiếng Việt ngắn gọn, tập trung vào bản chất kèm code ví dụ trực diện, không nói triết lý dài dòng". Yêu cầu "tiếng Việt ngắn gọn" giúp cố định ngôn ngữ phản hồi ngay cả khi gặp thuật ngữ kỹ thuật, đồng thời ghìm chặt độ dài đầu ra để tiết kiệm token và giảm độ trễ khi stream; cụm từ "tập trung vào bản chất kèm code ví dụ trực diện" buộc mô hình bỏ qua những lời chào hỏi xã giao rườm rà, đưa ra hướng giải quyết cụ thể để người học có thể đọc hiểu và áp dụng ngay vào bài code.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý là bộ nhớ ngữ cảnh bị cắt cứng ở 3 lượt gần nhất, khiến mô hình lập tức quên các ràng buộc ban đầu khi người dùng trao đổi dài. Để khắc phục, tôi đề xuất triển khai cơ chế tóm tắt ngữ cảnh tự động: khi hội thoại vượt quá 3 lượt, hệ thống gọi một mô hình nhẹ như Flash hoặc Mini để gom các lượt chat cũ thành một đoạn tóm tắt ngắn nạp thẳng vào system prompt, vừa giúp trợ lý ghi nhớ thông tin xuyên suốt phiên làm việc vừa giữ chi phí token ở mức tối thiểu.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
