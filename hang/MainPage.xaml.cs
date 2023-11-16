using CommunityToolkit.Maui.Storage;
using Plugin.Maui.Audio;
using System.Xml.Schema;
using System.IO;
using Microsoft.Maui.Animations;
//using Plugin.Maui.AudioRecorder.Abstractions;


namespace hang
{


    public partial class MainPage : ContentPage
    {

      
        readonly IAudioManager _audioManager;
        readonly IAudioRecorder _audioRecorder;
        public MainPage(IAudioManager audioManager)
        {
            InitializeComponent();

            _audioManager = audioManager;
            _audioRecorder = audioManager.CreateRecorder();
          

        }

        private bool isSquare = false;
        private bool isImage = false;

        private async void OnCounterClicked(object sender, EventArgs e)
        {

            if (isImage)
            {
                Kezdo.Source = "smile.png";
                isImage = false;
            }
            else
            {
                Kezdo.Source = "ani.gif";
                Kezdo.IsAnimationPlaying = true;
                isImage = true;
            }



            if (isSquare)
            {
                CounterBtn.Text = "Hangfelvétel";
                CounterBtn.CornerRadius = 60;
                CounterBtn.WidthRequest = 110;
                CounterBtn.HeightRequest = 110;
                CounterBtn.Margin = 10;
                isSquare = false;

            }
            else
            {
                CounterBtn.Text = "Állj";
                CounterBtn.CornerRadius = 0;
                CounterBtn.WidthRequest = 90;
                CounterBtn.HeightRequest = 90;
                CounterBtn.Margin = 20;
                isSquare = true;
            }

            if (await Permissions.RequestAsync<Permissions.Microphone>() != PermissionStatus.Granted)
            {
                return;
                
            }

            if (!_audioRecorder.IsRecording)
            {
                await _audioRecorder.StartAsync();
            }
            else
            {

                var recordedAudio = await _audioRecorder.StopAsync();

                                            
                //lejátszás
                 var player = AudioManager.Current.CreatePlayer(recordedAudio.GetAudioStream());
                 player.Play();
               


            }


        }


     



    }
}