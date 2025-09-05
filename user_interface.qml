import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15

ApplicationWindow {
    visible: true
    width: 500
    height: 400
    title: "Music Learning App"

    Column {
        anchors.centerIn: parent
        spacing: 20

        Text {
            text: "Listen to the teacher's audio and then record your own."
            wrapMode: Text.Wrap
            font.pointSize: 14
            horizontalAlignment: Text.AlignHCenter
            width: parent.width * 0.9
        }

        // Teacher Audio Player
        Row {
            spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter

            Button {
                text: "▶ Play Teacher Audio"
                onClicked: {
                    backend.playTeacherAudio()  // Python function
                }
            }

            Button {
                text: "⏸ Pause"
                onClicked: {
                    backend.pauseTeacherAudio()
                }
            }
        }

        // Recording Controls
        Row {
            spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter

            Button {
                text: "🎤 Start Recording"
                onClicked: {
                    backend.startRecording()
                }
            }

            Button {
                text: "⏹ Stop Recording"
                onClicked: {
                    backend.stopRecording()
                }
            }
        }

        // Submit Recording
        Button {
            text: "✅ Submit"
            anchors.horizontalCenter: parent.horizontalCenter
            onClicked: {
                backend.processRecording()  // Runs your Python backend
            }
        }

        // Result Area
        Text {
            id: resultLabel
            text: "Mistakes: (waiting for analysis...)"
            wrapMode: Text.Wrap
            font.pointSize: 12
            horizontalAlignment: Text.AlignHCenter
            width: parent.width * 0.9
        }

        Image {
            id: pitchPlot
            width: 400
            height: 200
            anchors.horizontalCenter: parent.horizontalCenter
            fillMode: Image.PreserveAspectFit
            source: ""   // Backend will update this path
        }
    }
}
