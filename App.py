import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import time
import pickle
from PIL import Image

class FaceRecognitionAttendance:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_ids = []
        self.known_face_names = []
        self.attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        self.next_id = 0
        self.load_database()
        
    def load_database(self):
        # Check if model exists
        if os.path.exists('face_model.yml'):
            self.recognizer.read('face_model.yml')
            
            # Load face names
            if os.path.exists('face_names.pkl'):
                with open('face_names.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_ids = data['ids']
                    self.known_face_names = data['names']
                    self.next_id = max(self.known_face_ids) + 1 if self.known_face_ids else 0
                st.success(f"Loaded {len(self.known_face_names)} faces from database")
            else:
                st.info("Face model exists but names data is missing")
        else:
            st.info("No existing face database found. You'll need to register faces first.")
        
        # Load attendance if exists
        if os.path.exists('attendance_log.csv'):
            self.attendance_df = pd.read_csv('attendance_log.csv')
    
    def save_database(self):
        # Save the model
        self.recognizer.write('face_model.yml')
        
        # Save the names
        with open('face_names.pkl', 'wb') as f:
            pickle.dump({
                'ids': self.known_face_ids,
                'names': self.known_face_names
            }, f)
        
    def save_attendance(self):
        self.attendance_df.to_csv('attendance_log.csv', index=False)
    
    def preprocess_face(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
            
        # Process first face (or the largest one)
        if len(faces) > 1:
            # Find the largest face
            largest_area = 0
            largest_face_idx = 0
            for i, (x, y, w, h) in enumerate(faces):
                if w*h > largest_area:
                    largest_area = w*h
                    largest_face_idx = i
            (x, y, w, h) = faces[largest_face_idx]
        else:
            (x, y, w, h) = faces[0]
            
        # Crop and normalize face
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (150, 150))
        
        return face_img, (x, y, w, h)
            
    def register_new_face(self, image, name):
        # Preprocess and detect face
        face_img, face_rect = self.preprocess_face(image)
        
        if face_img is None:
            return False, "No face detected in the image. Please try again."
        
        # Check if name already exists
        if name in self.known_face_names:
            return False, f"A person with the name '{name}' is already registered. Please use a different name."
        
        # Create training data
        face_id = self.next_id
        
        # Train the recognizer with this new face
        self.recognizer.update([face_img], np.array([face_id]))
        
        # Add to database
        self.known_face_ids.append(face_id)
        self.known_face_names.append(name)
        self.next_id += 1
        
        # Save the updated database
        self.save_database()
        
        return True, f"Successfully registered {name}"
    
    def mark_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if already marked attendance today
        today_attendance = self.attendance_df[self.attendance_df['Date'] == date_str]
        if name in today_attendance['Name'].values:
            return False, f"{name} has already been marked present today"
        
        # Add to attendance log
        new_row = pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})
        self.attendance_df = pd.concat([self.attendance_df, new_row], ignore_index=True)
        
        # Save attendance
        self.save_attendance()
        
        return True, f"Marked attendance for {name}"
    
    def process_frame(self, frame):
        # Make a copy to avoid modifying the original
        display_frame = frame.copy()
        
        # Preprocess and detect face
        face_img, face_rect = self.preprocess_face(frame)
        
        recognized_names = []
        
        if face_img is not None and face_rect is not None:
            x, y, w, h = face_rect
            
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Only try to recognize if we have registered faces
            if self.known_face_ids:
                # Predict
                face_id, confidence = self.recognizer.predict(face_img)
                
                # Lower confidence means better match in LBPH
                if confidence < 80:  # Threshold for recognition
                    # Find the name
                    idx = self.known_face_ids.index(face_id)
                    name = self.known_face_names[idx]
                    recognized_names.append(name)
                    
                    # Display name and confidence
                    text = f"{name} ({100-confidence:.1f}%)"
                    cv2.putText(display_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Unknown face
                    cv2.putText(display_frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return display_frame, recognized_names

def main():
    st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
    
    st.title("Face Recognition Attendance System")
    st.markdown("#### Automated attendance tracking using facial recognition")
    
    # Initialize system
    system = FaceRecognitionAttendance()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 Take Attendance", "👤 Register New Face", "📋 Attendance Records"])
    
    with tab1:
        st.header("Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Registered Faces", len(system.known_face_names))
            
            if not system.attendance_df.empty:
                today = datetime.now().strftime("%Y-%m-%d")
                today_attendance = system.attendance_df[system.attendance_df['Date'] == today]
                st.metric("Today's Attendance", len(today_attendance))
                
                # Attendance rate
                if system.known_face_names:
                    attendance_rate = (len(today_attendance) / len(system.known_face_names)) * 100
                    st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
        
        with col2:
            if not system.attendance_df.empty:
                # Show recent attendance
                st.subheader("Recent Attendance")
                recent = system.attendance_df.sort_values(by='Date', ascending=False).head(5)
                st.dataframe(recent)
        
        if not system.attendance_df.empty:
            # Show attendance trends
            st.subheader("Attendance Trends")
            attendance_counts = system.attendance_df.groupby('Date').size().reset_index(name='Count')
            st.line_chart(attendance_counts.set_index('Date'))
    
    with tab2:
        st.header("Take Attendance")
        
        if len(system.known_face_names) == 0:
            st.warning("No faces registered in the database. Please register faces first.")
        else:
            st.info("Click 'Start Camera' to begin taking attendance. The system will automatically recognize registered faces.")
            
            start_camera = st.button("Start Camera")
            
            if start_camera:
                # Create a placeholder for the webcam feed
                video_placeholder = st.empty()
                status_placeholder = st.empty()
                
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera connection.")
                else:
                    attendance_marked = []
                    
                    # Add a stop button
                    stop_button_col = st.columns(3)
                    stop = stop_button_col[1].button("Stop Camera", key="stop_camera")
                    
                    while not stop:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Failed to capture frame from camera")
                            break
                        
                        # Process frame
                        display_frame, recognized_names = system.process_frame(frame)
                        
                        # Mark attendance for recognized faces
                        for name in recognized_names:
                            if name not in attendance_marked:
                                success, message = system.mark_attendance(name)
                                if success:
                                    attendance_marked.append(name)
                                    status_placeholder.success(message)
                        
                        # Display the frame
                        video_placeholder.image(display_frame, channels="BGR", use_column_width=True)
                        
                        # Update every 100ms
                        time.sleep(0.1)
                    
                    # Release the camera
                    cap.release()
    
    # Here's the fixed version of the camera capture section in the Register New Face tab

        with tab3:
            st.header("Register New Face")
    st.markdown("Add new people to the facial recognition database")
    
    registration_option = st.radio("Registration Method", ["Upload Image", "Capture from Camera"])
    
    name = st.text_input("Person's Name")
    
    if registration_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None and name:
            # Convert file to image
            image = np.array(Image.open(uploaded_file))
            
            # Convert RGB to BGR (OpenCV format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            st.image(image, caption="Uploaded Image", channels="BGR", use_column_width=True)
            
            if st.button("Register Face"):
                success, message = system.register_new_face(image, name)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                    
    else:  # Capture from Camera
        if name:
            st.info("Click 'Capture' to take a snapshot when ready")
            
            # Use session state to track capture state
            if 'captured_image' not in st.session_state:
                st.session_state.captured_image = None
            
            capture_btn = st.button("Capture")
            
            if capture_btn:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera connection.")
                else:
                    # Display live feed
                    preview_placeholder = st.empty()
                    
                    for _ in range(10):  # Show a few frames so user can position
                        ret, frame = cap.read()
                        if ret:
                            preview_placeholder.image(frame, channels="BGR", caption="Camera Feed", use_column_width=True)
                            time.sleep(0.2)
                    
                    # Capture final frame
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        # Store in session state
                        st.session_state.captured_image = frame.copy()
                        preview_placeholder.image(frame, channels="BGR", caption="Captured Image", use_column_width=True)
                    else:
                        st.error("Failed to capture image from camera")
            
            # Show the captured image from session state
            if st.session_state.captured_image is not None:
                st.image(st.session_state.captured_image, channels="BGR", caption="Captured Image", use_column_width=True)
                
                register_btn = st.button("Register Face", key="register_captured")
                if register_btn:
                    success, message = system.register_new_face(st.session_state.captured_image, name)
                    if success:
                        st.success(message)
                        # Clear image after successful registration
                        st.session_state.captured_image = None
                    else:
                        st.error(message)
        else:
            st.warning("Please enter a name first")
    
    with tab4:
        st.header("Attendance Records")
        
        if system.attendance_df.empty:
            st.info("No attendance records yet")
        else:
            # Filter options
            st.subheader("Filter Records")
            
            col1, col2 = st.columns(2)
            
            with col1:
                unique_dates = sorted(system.attendance_df['Date'].unique(), reverse=True)
                selected_date = st.selectbox("Select Date", ["All"] + list(unique_dates))
            
            with col2:
                unique_names = sorted(system.attendance_df['Name'].unique())
                selected_name = st.selectbox("Select Person", ["All"] + list(unique_names))
            
            # Apply filters
            filtered_df = system.attendance_df.copy()
            
            if selected_date != "All":
                filtered_df = filtered_df[filtered_df['Date'] == selected_date]
                
            if selected_name != "All":
                filtered_df = filtered_df[filtered_df['Name'] == selected_name]
            
            # Display records
            st.subheader("Attendance Records")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export option
            if st.button("Export to CSV"):
                filtered_df.to_csv("exported_attendance.csv", index=False)
                st.success("Exported to exported_attendance.csv")
                
            # Statistics
            st.subheader("Statistics")
            
            if not filtered_df.empty:
                # Attendance by date
                date_counts = filtered_df.groupby('Date').size().reset_index(name='Count')
                st.bar_chart(date_counts.set_index('Date'))
                
                # Attendance by person
                if len(filtered_df['Name'].unique()) > 1:
                    person_counts = filtered_df.groupby('Name').size().reset_index(name='Count')
                    st.bar_chart(person_counts.set_index('Name'))

if __name__ == '__main__':
    main()


                