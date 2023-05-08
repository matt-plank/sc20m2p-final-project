import "./app.css";
import ImageBox from "./components/imageBox/imageBox";

function App() {
  return (
    <div className="image-wrapper">
      <ImageBox canEncode={true} />
      <ImageBox canEncode={true} />
      <button className="btn">Combine</button>
      <ImageBox />
    </div>
  );
}

export default App;
