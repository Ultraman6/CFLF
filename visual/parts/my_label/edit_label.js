export default {
  body: `

              
  `,
  data() {
    return {
      value: 0,
    };
  },
  methods: {
    handle_click() {
      this.value += 1;
      this.$emit("change", this.value);
    },
    select_type(){

    },
    select_spot(){

    },
    select_algo(){

    },
    delete(){

    },
    reset() {
      this.value = 0;
    },
  },
  props: {
      algo_type_options: Array,
      algo_spot_options: Array,
      algo_name_options: Array,
      selected_result: Object
  },
};