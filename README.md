# Time-Series-Forecasting-Model-Using-Python
Sử Dụng Mô Hình Dự Báo Chuỗi Thời Gian Bằng Python
Mô hình LSTM
Mô hình LSTM (Long Short-Term Memory) là một dạng đặc biệt của mạng nơ-ron hồi quy (RNN) được thiết kế để xử lý dữ liệu chuỗi và giải quyết vấn đề vanishing gradient trong RNN.
Một LSTM có cấu trúc tương tự như RNN, nhưng nó có thêm các cơ chế đặc biệt như cổng (gates) để điều chỉnh thông tin được lưu trữ và truyền qua lại trong quá trình tính toán.
Các cổng trong LSTM bao gồm:
	 Cổng quên (Forget gate): Xác định thông tin nào trong trạng thái trước đó cần bị lãng quên. Nó quyết định giữ lại thông tin nào và bỏ qua thông tin nào từ quá khứ.
	 Cổng đầu vào (Input gate): Xác định thông tin mới nào sẽ được lưu trữ trong trạng thái tiếp theo. Nó quyết định thông tin mới nào sẽ được cập nhật từ đầu vào hiện tại.
	 Cổng đầu ra (Output gate): Xác định phần nào của trạng thái hiện tại sẽ là đầu ra của LSTM. Nó quyết định thông tin nào sẽ được đưa ra từ trạng thái hiện tại.
	 Trạng thái ẩn (Cell state): Là một bộ nhớ dài hạn, lưu trữ thông tin từ quá khứ và truyền thông tin tới các bước tính toán tiếp theo trong chuỗi.
	 Với các cổng này, LSTM có khả năng lưu trữ thông tin quan trọng từ quá khứ và loại bỏ thông tin không quan trọng. Điều này giúp LSTM giải quyết vấn đề vanishing gradient và có khả năng xử lý được các chuỗi dài và tương quan phức tạp hơn so với RNN thông thường.
	 Mô hình LSTM được sử dụng rộng rãi trong các lĩnh vực như xử lý ngôn ngữ tự nhiên, dịch máy, nhận dạng giọng nói, dự báo chuỗi thời gian và nhiều ứng dụng khác liên quan đến dữ liệu chuỗi.
Tóm lại, mô hình LSTM là một dạng đặc biệt của mạng nơ-ron hồi quy, được thiết kế để xử lý dữ liệu chuỗi và giải quyết vấn đề vanishing gradient. Nó sử dụng các cổng để điều chỉnh thông tin và trạng thái trong quá trình tính toán, giúp nó lưu trữ thông tin quan trọng và loại bỏ thông tin không quan trọng. LSTM đã được chứng minh là một công cụ mạnh mẽ trong việc xử lý và dự đoán các dữ liệu chuỗi phức tạp.
 
Hình 1 4 Cấu trúc mô hình LSTM

Output:ct,ht,ta gọi c là cell state,t là hidden state.
Input: ct-1,ht-1,xt.Trong đó xt là input ở state thứ t của model.
         Đọc biểu đồ ở trên ta có thể thấy kí hiệu σ, tanh có nghĩa là ở bước đấy dùng hàm sigma , tanh.
Phép nhân ở đây là phép nhân từng phần tử, phép cộng là cộng ma trận.
Ft,it,ot tương ứng cới forget gate,input gate,output gate:
Forget gate: ft=σ(U𝑓*x𝑡+W𝑓*h𝑡−1+b𝑓)
         Input gate: it=σ(U𝑖*x𝑡+W𝑖*h𝑡−1+bi)
         Output gate: ot= σ(Uo*xt+Wo*ht-1+bo)
Nhận xét 0<ft,it,ot<1;bf,bi,bo là các hệ số bias,hệ số W,U giống như bài RNN 
c = tanh⁡(Uc * xt+Wc*ht-1+bc)
ct = ft*ct-1+it*ct, forget gate quyết định xem lấy bao nhiêu từ cell state trước và input gate sẽ quyết định lấy bao nhiêu từ input state và hidden layer của layer trước.
 ht = ot*tanh⁡(ct) ,output gate quyết định xem cần lấy bao nhiêu từ cell state để trở thành output của hidden state. Ngoài ra ht cũng đc dùng để tính ra output yt cho state t.
Nhận xét: ht,ct khá giống với RNN,nên model có short tẻm memory.Trong khi đó ct giống như một băng chuyền ở trên mô hình RNN, thông tin nào cần quan trọng và dùng ở sau sẽ được gửi vào dùng khi cần=>có thể mang thông tin từ đi xa.Do đó mô hình LSTM có cả short term memory và long term memory.
