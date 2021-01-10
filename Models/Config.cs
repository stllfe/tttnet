namespace TTT{
    public static class Config
    {
        public const int RANDOM_SEED = 333;
        public const int SIDE_SIZE = 4;
        public const int WEIGHTS_PRECISION = 7;
        public const float MAX_INIT_WEIGHTS = 0.001f;
        public const float EPS = 0.00001f;
        public const float LEARNING_RATE = 0.15f;
        public const int NUM_EPOCHS = 1000;
        public const int LOG_EVERY = NUM_EPOCHS;
        public const int NUM_HIDDEN_LAYERS = 1;
        public static int[] HIDDEN_LAYERS_SIZES = new int[]{8};
    }
}